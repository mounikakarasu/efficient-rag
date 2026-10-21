from __future__ import annotations

import functools

from sentence_transformers import CrossEncoder

from app.config import config
from app.embeddings import resolve_device
from app.logging_conf import get_logger

log = get_logger("app.rerank")


@functools.lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    device = resolve_device()
    log.info("loading reranker", extra={"model": config.reranker_model, "device": device})
    return CrossEncoder(config.reranker_model, device=device, max_length=512)


def rerank(query: str, candidates: list[tuple[int, str]], top_k: int) -> list[tuple[int, float]]:
    if not candidates:
        return []
    pairs = [[query, text] for _idx, text in candidates]
    scores = get_reranker().predict(pairs, show_progress_bar=False)
    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
    return [(candidates[i][0], float(scores[i])) for i in order]
