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
from app.qdrant_store import get_client
from app.qdrant_store import search as qsearch
from app.vector_index import build_flat, build_hnsw, load_chunk_meta, load_vectors
from app.vector_index import search as faiss_search

log = get_logger("bench.qdrant")
ROOT = Path(__file__).resolve().parent.parent


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _recall(approx: list[list[int]], exact: list[list[int]]) -> float:
    total = 0.0
    for a, e in zip(approx, exact):
        es = set(int(x) for x in e if x >= 0)
        if not es:
            continue
        total += sum(1 for x in a if int(x) in es) / len(es)
    return total / len(approx) if approx else 0.0


def load_query_vectors(eval_path: Path, limit: int) -> np.ndarray:
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return embed_texts([r["query"] for r in rows][:limit])


def run(eval_path: Path, k: int, query_limit: int) -> dict:
    configure_logging("WARNING")
    vectors = load_vectors()
    queries = load_query_vectors(eval_path, query_limit)

    flat = build_flat(vectors)
    exact = [list(row) for row in faiss_search(flat, queries, k)]

    hnsw = build_hnsw(vectors, m=32, ef_construction=200, ef_search=64)
    faiss_search(hnsw, queries[:5], k)
    faiss_ids, faiss_lat = [], []
    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        ids = faiss_search(hnsw, queries[i : i + 1], k)
        faiss_lat.append((time.perf_counter() - t0) * 1000)
        faiss_ids.append(list(ids[0]))

    client = get_client()
    qsearch(client, config.qdrant_collection, queries[0], k)
    qdrant_ids, qdrant_lat = [], []
    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        points = qsearch(client, config.qdrant_collection, queries[i], k)
        qdrant_lat.append((time.perf_counter() - t0) * 1000)
        qdrant_ids.append([int(p.id) for p in points])

    meta = load_chunk_meta()
    sample_doc = meta[int(exact[0][0])].get("doc_id")
    filtered = qsearch(client, config.qdrant_collection, queries[0], k, doc_id=sample_doc)
    filter_ok = all((p.payload or {}).get("doc_id") == sample_doc for p in filtered)

    return {
        "vectors": int(vectors.shape[0]),
        "queries": int(queries.shape[0]),
        "k": k,
        "rows": [
            {
                "backend": "FAISS HNSW (in-process)",
                "recall_at_k": round(_recall(faiss_ids, exact), 4),
                "latency_ms_p50": round(_pct(faiss_lat, 50), 3),
                "latency_ms_p95": round(_pct(faiss_lat, 95), 3),
            },
            {
                "backend": "Qdrant HNSW (service)",
                "recall_at_k": round(_recall(qdrant_ids, exact), 4),
                "latency_ms_p50": round(_pct(qdrant_lat, 50), 3),
                "latency_ms_p95": round(_pct(qdrant_lat, 95), 3),
            },
        ],
        "filter_doc_id": sample_doc,
        "filter_returned": len(filtered),
        "filter_ok": bool(filter_ok),
    }


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "qdrant_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "recall_at_k", "latency_ms_p50", "latency_ms_p95"])
        writer.writeheader()
        writer.writerows(report["rows"])

    lines = [
        "# FAISS vs Qdrant",
        "",
        f"_{report['vectors']:,} vectors, {report['queries']} real queries, "
        f"recall@{report['k']} vs exact search. Both use HNSW (m=32)._",
        "",
        f"| Backend | recall@{report['k']} | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(f"| {r['backend']} | {r['recall_at_k']} | {r['latency_ms_p50']} | {r['latency_ms_p95']} |")
    lines += [
        "",
        "## Metadata filtering (Qdrant only)",
        "",
        f"A filtered search for doc_id `{report['filter_doc_id']}` returned "
        f"{report['filter_returned']} results, all from that document: "
        f"{'passed' if report['filter_ok'] else 'FAILED'}. FAISS has no native payload "
        "filtering; Qdrant applies it inside the search.",
        "",
        "FAISS runs in-process (no network hop); Qdrant runs as a service and adds "
        "persistence, concurrency, payload filtering, and built-in quantization at the "
        "cost of a localhost round-trip.",
        "",
    ]
    (out_dir / "qdrant_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare FAISS and Qdrant retrieval.")
    parser.add_argument("--eval", type=Path, default=config.eval_dir / "qa_eval.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    report = run(args.eval, args.k, args.queries)
    write_report(report, args.out)
    for r in report["rows"]:
        print(f"{r['backend']:26s} recall@{args.k}={r['recall_at_k']} p50={r['latency_ms_p50']}ms")
    print(f"filter doc_id={report['filter_doc_id']} returned={report['filter_returned']} ok={report['filter_ok']}")
    print(f"Wrote {args.out / 'qdrant_benchmark.md'}")


if __name__ == "__main__":
    main()
