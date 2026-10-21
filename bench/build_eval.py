from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.config import config
from app.logging_conf import configure_logging, get_logger

log = get_logger("bench.eval")


def load_answers(answers_path: Path) -> dict:
    mapping = {}
    for line in answers_path.open(encoding="utf-8"):
        rec = json.loads(line)
        mapping[rec["id"]] = rec["accepted_answer"]
    return mapping


def build_eval(corpus_dir: Path, out_dir: Path, size: int, seed: int = 13) -> dict:
    corpus_path = corpus_dir / "corpus.jsonl"
    answers = load_answers(corpus_dir / "answers.jsonl")
    docs = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
    usable = [d for d in docs if answers.get(d["id"]) and len(d["title"]) > 15]

    rng = random.Random(seed)
    sample = rng.sample(usable, min(size, len(usable)))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in sample:
            rec = {
                "qid": d["id"],
                "query": d["title"],
                "relevant_doc_id": d["id"],
                "ground_truth_answer": answers[d["id"]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {"eval_questions": len(sample), "path": str(out_path)}
    log.info("eval set built", extra=stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Build a held-out QA eval set from the corpus.")
    parser.add_argument("--corpus", type=Path, default=config.corpus_dir)
    parser.add_argument("--out", type=Path, default=config.eval_dir)
    parser.add_argument("--size", type=int, default=config.eval_size)
    args = parser.parse_args(argv)

    stats = build_eval(args.corpus, args.out, args.size)
    print(f"Eval set: {stats['eval_questions']} questions -> {stats['path']}")


if __name__ == "__main__":
    main()
