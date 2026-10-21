from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import faiss

from app.config import Config, config
from app.embeddings import embed_texts
from app.logging_conf import configure_logging, get_logger, new_correlation_id

log = get_logger("app.ingest")

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def iter_documents(src: Path) -> Iterator[tuple[Path, str]]:
    for path in sorted(src.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = load_text(path)
            if text.strip():
                yield path.relative_to(src), text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = text.strip()
    n = len(text)
    if n == 0:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            ws = text.rfind(" ", start + overlap, end)
            if ws != -1 and ws > start:
                end = ws
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        if end >= n:
            break
        new_start = end - overlap
        start = new_start if new_start > start else end
    return chunks


def collect_chunks(src: Path, cfg: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rel_path, text in iter_documents(src):
        source = rel_path.as_posix()
        for ci, (piece, cs, ce) in enumerate(
            chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    id=len(chunks),
                    text=piece,
                    source=source,
                    chunk_index=ci,
                    char_start=cs,
                    char_end=ce,
                )
            )
    return chunks


def build_index(src: Path, index_dir: Path, cfg: Config) -> dict:
    cid = new_correlation_id()
    t0 = time.perf_counter()

    chunks = collect_chunks(src, cfg)
    if not chunks:
        raise SystemExit(
            f"No supported documents found under {src} "
            f"(looked for {sorted(SUPPORTED_SUFFIXES)})."
        )

    log.info("embedding chunks", extra={"chunks": len(chunks), "src": str(src)})
    embeddings = embed_texts([c.text for c in chunks])
    dim = int(embeddings.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.jsonl"
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    build_seconds = time.perf_counter() - t0
    stats = {
        "documents": len({c.source for c in chunks}),
        "chunks": len(chunks),
        "embedding_model": cfg.embedding_model,
        "embedding_dim": dim,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "index_build_seconds": round(build_seconds, 3),
    }
    (index_dir / "index_meta.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    log.info("index built", extra=stats)
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from documents.")
    parser.add_argument("--src", type=Path, default=config.data_dir)
    parser.add_argument("--index", type=Path, default=config.index_dir)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    args = _parse_args(argv)
    stats = build_index(args.src, args.index, config)
    print(
        f"Indexed {stats['chunks']} chunks from {stats['documents']} document(s) "
        f"in {stats['index_build_seconds']}s -> {args.index}"
    )


if __name__ == "__main__":
    main()
