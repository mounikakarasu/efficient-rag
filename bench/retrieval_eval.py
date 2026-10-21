from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from app.config import config
from app.hybrid import HybridRetriever
from app.logging_conf import configure_logging, get_logger
from app.retriever import AnnRetriever
from app.vector_index import load_chunk_meta

log = get_logger("bench.retrieval_eval")
ROOT = Path(__file__).resolve().parent.parent


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def load_eval(eval_path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def evaluate(name: str, retriever, rows: list[dict], meta: list[dict], k: int) -> dict:
    hits = 0
    latencies = []
    for row in rows:
        t0 = time.perf_counter()
        results = retriever.search(row["query"], top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        doc_ids = {meta[r.id].get("doc_id") for r in results}
        if row["relevant_doc_id"] in doc_ids:
            hits += 1
    return {
        "method": name,
        "recall_at_k": round(hits / len(rows), 4) if rows else 0.0,
        "latency_ms_p50": round(_pct(latencies, 50), 1),
        "latency_ms_p95": round(_pct(latencies, 95), 1),
    }


def run(eval_path: Path, k: int, query_limit: int) -> dict:
    configure_logging("WARNING")
    rows = load_eval(eval_path, query_limit)
    meta = load_chunk_meta()

    dense = AnnRetriever.load()
    hybrid = HybridRetriever.load(rerank_enabled=False)
    hybrid_rerank = HybridRetriever.load(rerank_enabled=True)

    results = [
        evaluate("Dense (HNSW)", dense, rows, meta, k),
        evaluate("Hybrid (BM25+dense+RRF)", hybrid, rows, meta, k),
        evaluate("Hybrid + cross-encoder rerank", hybrid_rerank, rows, meta, k),
    ]
    return {"queries": len(rows), "k": k, "results": results}


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "retrieval_eval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "recall_at_k", "latency_ms_p50", "latency_ms_p95"])
        writer.writeheader()
        writer.writerows(report["results"])

    lines = [
        "# Retrieval Quality",
        "",
        f"_Recall@{report['k']} against {report['queries']} labeled Stack Overflow queries "
        "(relevant doc = the question the query was drawn from)._",
        "",
        f"| Method | recall@{report['k']} | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(f"| {r['method']} | {r['recall_at_k']} | {r['latency_ms_p50']} | {r['latency_ms_p95']} |")
    lines.append("")
    (out_dir / "retrieval_eval.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (dense vs hybrid vs rerank).")
    parser.add_argument("--eval", type=Path, default=config.eval_dir / "qa_eval.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    report = run(args.eval, args.k, args.queries)
    write_report(report, args.out)
    for r in report["results"]:
        print(f"{r['method']:34s} recall@{args.k}={r['recall_at_k']} p50={r['latency_ms_p50']}ms")
    print(f"Wrote {args.out / 'retrieval_eval.md'}")


if __name__ == "__main__":
    main()
