from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from app.config import config
from app.embeddings import embed_texts
from app.logging_conf import configure_logging, get_logger
from app.vector_index import (
    build_flat,
    build_hnsw,
    build_ivfpq,
    index_num_bytes,
    load_vectors,
    search,
)

log = get_logger("bench.ann")
ROOT = Path(__file__).resolve().parent.parent


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    total = 0.0
    for approx_row, exact_row in zip(approx_ids, exact_ids):
        exact_set = set(int(x) for x in exact_row if x >= 0)
        if not exact_set:
            continue
        hit = sum(1 for x in approx_row if int(x) in exact_set)
        total += hit / len(exact_set)
    return total / len(approx_ids) if len(approx_ids) else 0.0


def _query_latencies_ms(index, queries: np.ndarray, k: int) -> list[float]:
    search(index, queries[:5], k)  # warm up
    latencies = []
    for i in range(queries.shape[0]):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        index.search(np.ascontiguousarray(q, dtype="float32"), k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def load_query_vectors(eval_path: Path, limit: int) -> np.ndarray:
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [r["query"] for r in rows][:limit]
    return embed_texts(queries)


def run(eval_path: Path, k: int, query_limit: int) -> dict:
    configure_logging("WARNING")
    vectors = load_vectors()
    n, dim = vectors.shape
    queries = load_query_vectors(eval_path, query_limit)
    log.info("ann benchmark start", extra={"vectors": n, "dim": dim, "queries": queries.shape[0]})

    t0 = time.perf_counter()
    flat = build_flat(vectors)
    flat_build = time.perf_counter() - t0
    exact_ids = search(flat, queries, k)

    configs = []

    def measure(name: str, index, build_seconds: float, params: str) -> dict:
        approx_ids = search(index, queries, k)
        latencies = _query_latencies_ms(index, queries, k)
        return {
            "index": name,
            "params": params,
            "recall_at_k": round(_recall_at_k(approx_ids, exact_ids), 4),
            "latency_ms_p50": round(_pct(latencies, 50), 3),
            "latency_ms_p95": round(_pct(latencies, 95), 3),
            "index_mb": round(index_num_bytes(index) / 1e6, 1),
            "build_seconds": round(build_seconds, 2),
        }

    configs.append(measure("Flat (exact)", flat, flat_build, "brute force"))

    t0 = time.perf_counter()
    hnsw = build_hnsw(vectors, m=32, ef_construction=200, ef_search=64)
    configs.append(measure("HNSW", hnsw, time.perf_counter() - t0, "M=32, efSearch=64"))

    t0 = time.perf_counter()
    ivfpq = build_ivfpq(vectors, nlist=2048, m=48, nbits=8, nprobe=16)
    ivfpq_build = time.perf_counter() - t0
    configs.append(measure("IVF-PQ", ivfpq, ivfpq_build, "nlist=2048, m=48, nprobe=16"))

    nprobe_sweep = []
    for nprobe in (4, 8, 16, 32, 64):
        ivfpq.nprobe = nprobe
        approx_ids = search(ivfpq, queries, k)
        latencies = _query_latencies_ms(ivfpq, queries, k)
        nprobe_sweep.append({
            "nprobe": nprobe,
            "recall_at_k": round(_recall_at_k(approx_ids, exact_ids), 4),
            "latency_ms_p50": round(_pct(latencies, 50), 3),
        })

    return {
        "vectors": n,
        "dim": dim,
        "queries": int(queries.shape[0]),
        "k": k,
        "configs": configs,
        "nprobe_sweep": nprobe_sweep,
    }


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ann_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "params", "recall_at_k", "latency_ms_p50",
                        "latency_ms_p95", "index_mb", "build_seconds"],
        )
        writer.writeheader()
        writer.writerows(report["configs"])

    flat = next(c for c in report["configs"] if c["index"].startswith("Flat"))
    lines = [
        "# ANN Benchmark",
        "",
        f"_{report['vectors']:,} vectors ({report['dim']}-dim) over {report['queries']} "
        f"real Stack Overflow queries; recall@{report['k']} measured against exact (Flat) search._",
        "",
        "| Index | Params | recall@{k} | p50 (ms) | p95 (ms) | Index size (MB) | Build (s) |".format(k=report["k"]),
        "|---|---|---|---|---|---|---|",
    ]
    for c in report["configs"]:
        lines.append(
            f"| {c['index']} | {c['params']} | {c['recall_at_k']} | {c['latency_ms_p50']} "
            f"| {c['latency_ms_p95']} | {c['index_mb']} | {c['build_seconds']} |"
        )
    lines += [
        "",
        f"Flat (exact) index is {flat['index_mb']} MB with p50 {flat['latency_ms_p50']} ms — "
        "the baseline ANN is compared against.",
        "",
        "## IVF-PQ nprobe sweep (recall vs latency trade-off)",
        "",
        f"| nprobe | recall@{report['k']} | p50 (ms) |",
        "|---|---|---|",
    ]
    for row in report["nprobe_sweep"]:
        lines.append(f"| {row['nprobe']} | {row['recall_at_k']} | {row['latency_ms_p50']} |")
    lines.append("")
    (out_dir / "ann_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark FAISS ANN indexes.")
    parser.add_argument("--eval", type=Path, default=config.eval_dir / "qa_eval.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    report = run(args.eval, args.k, args.queries)
    write_report(report, args.out)
    for c in report["configs"]:
        print(f"{c['index']:14s} recall@{args.k}={c['recall_at_k']} "
              f"p50={c['latency_ms_p50']}ms size={c['index_mb']}MB")
    print(f"Wrote {args.out / 'ann_benchmark.md'}")


if __name__ == "__main__":
    main()
