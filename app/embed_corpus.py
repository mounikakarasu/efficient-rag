from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from app.config import Config, config
from app.embeddings import embed_texts, embedding_dim, get_embedder, resolve_device
from app.ingest import chunk_text
from app.logging_conf import configure_logging, get_logger

log = get_logger("app.embed_corpus")


def iter_chunks(corpus_path: Path, chunk_size: int, overlap: int) -> Iterator[dict]:
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            for ci, (piece, _cs, _ce) in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
                yield {
                    "doc_id": doc["id"],
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "chunk_index": ci,
                    "text": piece,
                }


def _reconcile(vectors_path: Path, chunks_path: Path, dim: int) -> int:
    if not vectors_path.exists() or not chunks_path.exists():
        return 0
    chunk_rows = sum(1 for _ in chunks_path.open(encoding="utf-8"))
    vec_rows = vectors_path.stat().st_size // (dim * 4)
    keep = min(chunk_rows, vec_rows)
    with vectors_path.open("r+b") as vf:
        vf.truncate(keep * dim * 4)
    if chunk_rows != keep:
        with chunks_path.open(encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f][:keep]
        chunks_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return keep


def batch_size_for(device: str, cfg: Config) -> int:
    return cfg.embed_batch_size_cuda if device == "cuda" else cfg.embed_batch_size


def embed_corpus(corpus_path: Path, out_dir: Path, cfg: Config, resume: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "vectors.f32"
    chunks_path = out_dir / "chunks.jsonl"

    get_embedder()  # warm up before timing
    dim = embedding_dim()
    device = resolve_device()
    batch_size = batch_size_for(device, cfg)
    skip = _reconcile(vectors_path, chunks_path, dim) if resume else 0
    if skip:
        log.info("resuming embedding", extra={"existing_chunks": skip})

    t0 = time.perf_counter()
    n_chunks = skip
    seen = 0
    buffer_texts: list[str] = []
    buffer_meta: list[dict] = []

    vmode = "ab" if skip else "wb"
    cmode = "a" if skip else "w"
    with vectors_path.open(vmode) as vf, chunks_path.open(cmode, encoding="utf-8") as cf:
        def flush() -> None:
            nonlocal n_chunks
            if not buffer_texts:
                return
            vectors = embed_texts(buffer_texts, batch_size=batch_size)
            vectors.tofile(vf)
            for meta in buffer_meta:
                cf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_chunks += len(buffer_texts)
            buffer_texts.clear()
            buffer_meta.clear()
            if n_chunks % (cfg.embed_flush_size * 5) == 0:
                log.info("embedding progress", extra={"chunks": n_chunks})

        for chunk in iter_chunks(corpus_path, cfg.chunk_size, cfg.chunk_overlap):
            seen += 1
            if seen <= skip:
                continue
            buffer_texts.append(chunk["text"])
            buffer_meta.append(chunk)
            if len(buffer_texts) >= cfg.embed_flush_size:
                flush()
        flush()

    elapsed = time.perf_counter() - t0
    embedded_now = n_chunks - skip
    backend = "torch" if device == "cuda" else cfg.embedding_backend
    stats = {
        "chunks": n_chunks,
        "embedded_this_run": embedded_now,
        "embedding_dim": dim,
        "elapsed_seconds": round(elapsed, 2),
        "chunks_per_second": round(embedded_now / elapsed, 1) if elapsed > 0 else 0.0,
        "vectors_mb": round(vectors_path.stat().st_size / 1e6, 1),
        "embedding_model": cfg.embedding_model,
        "device": device,
        "backend": backend,
        "batch_size": batch_size,
    }
    (out_dir / "embed_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("embedding complete", extra=stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Embed a corpus into a vector store.")
    parser.add_argument("--corpus", type=Path, default=config.corpus_dir / "corpus.jsonl")
    parser.add_argument("--out", type=Path, default=config.embeddings_dir)
    args = parser.parse_args(argv)

    stats = embed_corpus(args.corpus, args.out, config)
    print(
        f"Embedded {stats['chunks']} chunks (dim {stats['embedding_dim']}) in "
        f"{stats['elapsed_seconds']}s = {stats['chunks_per_second']} chunks/s; "
        f"{stats['vectors_mb']} MB -> {args.out}"
    )


if __name__ == "__main__":
    main()
