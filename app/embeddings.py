from __future__ import annotations

import functools
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import config
from app.logging_conf import get_logger

log = get_logger("app.embeddings")


def resolve_device(preference: str | None = None) -> str:
    preference = preference or config.embedding_device
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@functools.lru_cache(maxsize=4)
def get_embedder(
    model_name: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    name = model_name or config.embedding_model
    device = resolve_device(device)
    backend = backend or config.embedding_backend
    if device == "cuda":
        backend = "torch"
    kwargs = {}
    if backend == "onnx":
        kwargs["model_kwargs"] = {"file_name": config.embedding_onnx_file}
    log.info("loading embedding model", extra={"model": name, "backend": backend, "device": device})
    return SentenceTransformer(name, device=device, backend=backend, **kwargs)


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    model = get_embedder()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


def embedding_dim() -> int:
    model = get_embedder()
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(fn())
