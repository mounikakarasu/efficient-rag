from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from app.config import config
from app.logging_conf import configure_logging, get_logger

log = get_logger("app.corpus")

_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    text = _TAG_RE.sub(" ", text or "")
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Document:
    id: str
    source: str
    url: str
    title: str
    text: str


def select_accepted_answer(answers: list[dict]) -> dict | None:
    if not answers:
        return None
    selected = [a for a in answers if str(a.get("selected", "")).lower() == "true"]
    pool = selected or answers

    def score(a: dict) -> float:
        try:
            return float(a.get("pm_score") or 0)
        except (TypeError, ValueError):
            return 0.0

    return max(pool, key=score)


def site_matches(record: dict, site_filter: str) -> bool:
    if not site_filter:
        return True
    needle = site_filter.lower()
    meta = record.get("metadata") or []
    return any(needle in (m or "").lower() for m in meta)


def normalize_record(record: dict) -> tuple[Document, str] | None:
    question = clean_html(record.get("question", ""))
    if not question:
        return None
    meta = record.get("metadata") or []
    url = meta[0] if meta else ""
    answers = record.get("answers") or []
    answer_text = "\n\n".join(clean_html(a.get("text", "")) for a in answers)
    text = question if not answer_text else f"{question}\n\n{answer_text}"
    title = " ".join(question.split())[:100]
    doc = Document(
        id=str(record.get("qid", "")),
        source="stackoverflow",
        url=url,
        title=title,
        text=text,
    )
    best = select_accepted_answer(answers)
    ground_truth = clean_html(best.get("text", "")) if best else ""
    return doc, ground_truth


def normalized_stream(
    records: Iterable[dict],
    site_filter: str,
    limit: int,
    skip: int = 0,
) -> Iterator[tuple[Document, str]]:
    produced = 0
    yielded = 0
    for record in records:
        if not site_matches(record, site_filter):
            continue
        norm = normalize_record(record)
        if norm is None:
            continue
        produced += 1
        if produced <= skip:
            continue
        yield norm
        yielded += 1
        if limit and yielded >= limit:
            return


def _load_stream(dataset_id: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split, streaming=True)


def build_corpus(
    out_dir: Path,
    dataset_id: str,
    split: str,
    site_filter: str,
    target_size: int,
    resume: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    answers_path = out_dir / "answers.jsonl"

    existing = 0
    if resume and corpus_path.exists():
        existing = sum(1 for _ in corpus_path.open(encoding="utf-8"))
    if existing >= target_size:
        log.info("corpus already complete", extra={"documents": existing})
        return {"documents": existing, "path": str(corpus_path)}

    remaining = target_size - existing
    mode = "a" if existing else "w"
    written = existing
    records = _load_stream(dataset_id, split)
    with corpus_path.open(mode, encoding="utf-8") as cf, answers_path.open(mode, encoding="utf-8") as af:
        for doc, ground_truth in normalized_stream(records, site_filter, remaining, skip=existing):
            cf.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
            af.write(json.dumps({"id": doc.id, "accepted_answer": ground_truth}, ensure_ascii=False) + "\n")
            written += 1
            if written % 250 == 0:
                log.info("corpus progress", extra={"documents": written})

    stats = {"documents": written, "path": str(corpus_path)}
    (out_dir / "corpus_meta.json").write_text(
        json.dumps(
            {"dataset_id": dataset_id, "split": split, "site_filter": site_filter, "documents": written},
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("corpus built", extra=stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Stream and build a document corpus.")
    parser.add_argument("--size", type=int, default=config.corpus_target_size)
    parser.add_argument("--out", type=Path, default=config.corpus_dir)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    stats = build_corpus(
        args.out,
        config.corpus_dataset_id,
        config.corpus_split,
        config.corpus_site_filter,
        args.size,
        resume=not args.no_resume,
    )
    print(f"Corpus: {stats['documents']} documents -> {stats['path']}")


if __name__ == "__main__":
    main()
