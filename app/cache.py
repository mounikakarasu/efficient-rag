from __future__ import annotations

import threading

import numpy as np

from app.config import config
from app.embeddings import embed_texts
from app.logging_conf import get_logger

log = get_logger("app.cache")


class SemanticCache:
    def __init__(self, threshold: float, max_size: int):
        self._threshold = threshold
        self._max_size = max_size
        self._vectors: list[np.ndarray] = []
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def lookup(self, query: str) -> tuple[dict | None, np.ndarray]:
        query_vec = embed_texts([query])[0]
        with self._lock:
            if not self._vectors:
                return None, query_vec
            sims = np.vstack(self._vectors) @ query_vec
            best = int(np.argmax(sims))
            if float(sims[best]) >= self._threshold:
                return self._entries[best], query_vec
        return None, query_vec

    def store(self, query_vec: np.ndarray, value: dict) -> None:
        with self._lock:
            self._vectors.append(query_vec)
            self._entries.append(value)
            if len(self._vectors) > self._max_size:
                self._vectors.pop(0)
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._entries.clear()


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(config.cache_threshold, config.cache_max_size)
    return _cache
