from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import config
from app.embeddings import embed_texts
from app.eval_metrics import (
    cosine,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l_f1,
    token_f1,
)
from app.logging_conf import configure_logging, get_logger
from app.rag import answer_question
from app.retriever import get_retriever
from app.vector_index import load_chunk_meta

log = get_logger("bench.rag_eval")
ROOT = Path(__file__).resolve().parent.parent


def load_eval(eval_path: Path) -> list[dict]:
    return [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _retrieved_doc_ids(results, meta) -> list:
    seen = []
    for r in results:
        doc_id = meta[r.id].get("doc_id")
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def evaluate_retrieval(retriever, rows, meta, k) -> dict:
    recalls, rrs, ndcgs = [], [], []
    for row in rows:
        ids = _retrieved_doc_ids(retriever.search(row["query"], top_k=k), meta)
        recalls.append(recall_at_k(ids, row["relevant_doc_id"], k))
        rrs.append(reciprocal_rank(ids, row["relevant_doc_id"]))
        ndcgs.append(ndcg_at_k(ids, row["relevant_doc_id"], k))
    return {
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(rrs)), 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
    }


def evaluate_generation(retriever, rows, k) -> dict:
    f1s, rouges, ans_sims, faiths = [], [], [], []
    for row in rows:
        answer = answer_question(row["query"], retriever=retriever, top_k=k)
        context = " ".join(s["text"] for s in answer.sources)
        f1s.append(token_f1(answer.answer, row["ground_truth_answer"]))
        rouges.append(rouge_l_f1(answer.answer, row["ground_truth_answer"]))
        vecs = embed_texts([answer.answer, row["ground_truth_answer"], context or " "])
        ans_sims.append(cosine(vecs[0], vecs[1]))
        faiths.append(cosine(vecs[0], vecs[2]))
    return {
        "answer_token_f1": round(float(np.mean(f1s)), 4),
        "answer_rouge_l": round(float(np.mean(rouges)), 4),
        "answer_similarity": round(float(np.mean(ans_sims)), 4),
        "faithfulness_proxy": round(float(np.mean(faiths)), 4),
    }


def run(eval_path: Path, k: int, ret_queries: int, gen_queries: int) -> dict:
    configure_logging("WARNING")
    rows = load_eval(eval_path)
    meta = load_chunk_meta()
    retriever = get_retriever()

    retrieval = evaluate_retrieval(retriever, rows[:ret_queries], meta, k)
    generation = evaluate_generation(retriever, rows[:gen_queries], k)
    return {
        "k": k,
        "retrieval_queries": min(ret_queries, len(rows)),
        "generation_queries": min(gen_queries, len(rows)),
        "retrieval": retrieval,
        "generation": generation,
    }


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ret, gen = report["retrieval"], report["generation"]
    rows = [{"metric": m, "value": v} for m, v in {**ret, **gen}.items()]
    with (out_dir / "rag_eval.csv").open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RAG Evaluation",
        "",
        "_Offline, judge-free metrics (no external API). Retrieval scored against "
        "labeled relevant docs; answers scored against ground-truth accepted answers._",
        "",
        f"## Retrieval (hybrid + rerank, {report['retrieval_queries']} queries)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| recall@{report['k']} | {ret['recall_at_k']} |",
        f"| MRR | {ret['mrr']} |",
        f"| nDCG@{report['k']} | {ret['ndcg_at_k']} |",
        "",
        f"## Generation ({report['generation_queries']} queries)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer token-F1 (vs ground truth) | {gen['answer_token_f1']} |",
        f"| Answer ROUGE-L (vs ground truth) | {gen['answer_rouge_l']} |",
        f"| Answer embedding similarity | {gen['answer_similarity']} |",
        f"| Faithfulness proxy (answer vs context) | {gen['faithfulness_proxy']} |",
        "",
    ]
    (out_dir / "rag_eval.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline RAG evaluation harness.")
    parser.add_argument("--eval", type=Path, default=config.eval_dir / "qa_eval.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ret-queries", type=int, default=100)
    parser.add_argument("--gen-queries", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    report = run(args.eval, args.k, args.ret_queries, args.gen_queries)
    write_report(report, args.out)
    print("Retrieval:", report["retrieval"])
    print("Generation:", report["generation"])
    print(f"Wrote {args.out / 'rag_eval.md'}")


if __name__ == "__main__":
    main()
