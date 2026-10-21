from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import config
from app.logging_conf import get_logger
from app.vector_index import load_chunk_meta

log = get_logger("app.sparse")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, bm25: BM25Okapi, size: int):
        self._bm25 = bm25
        self.size = size

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        tokenized = [tokenize(t) for t in texts]
        return cls(BM25Okapi(tokenized), len(texts))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"bm25": self._bm25, "size": self.size}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["size"])


def build_bm25(embeddings_dir: Path | None = None) -> dict:
    embeddings_dir = embeddings_dir or config.embeddings_dir
    meta = load_chunk_meta(embeddings_dir)
    index = BM25Index.build([m["text"] for m in meta])
    path = embeddings_dir / "bm25.pkl"
    index.save(path)
    stats = {"documents": index.size, "path": str(path)}
    log.info("bm25 index built", extra=stats)
    return stats


def main() -> None:
    from app.logging_conf import configure_logging

    configure_logging(config.log_level)
    stats = build_bm25()
    print(f"BM25 index over {stats['documents']} chunks -> {stats['path']}")


if __name__ == "__main__":
    main()
