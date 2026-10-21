from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.config import config
from app.logging_conf import get_logger

log = get_logger("app.vector_index")


def load_vectors(embeddings_dir: Path | None = None, dim: int | None = None) -> np.ndarray:
    embeddings_dir = embeddings_dir or config.embeddings_dir
    dim = dim or 384
    raw = np.fromfile(embeddings_dir / "vectors.f32", dtype="float32")
    n = raw.size // dim
    return np.ascontiguousarray(raw[: n * dim].reshape(n, dim))


def load_chunk_meta(embeddings_dir: Path | None = None) -> list[dict]:
    embeddings_dir = embeddings_dir or config.embeddings_dir
    path = embeddings_dir / "chunks.jsonl"
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(json.loads(line))
    return out


def build_flat(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(
    vectors: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    index.hnsw.efSearch = ef_search
    return index


def build_ivfpq(
    vectors: np.ndarray,
    nlist: int = 2048,
    m: int = 48,
    nbits: int = 8,
    nprobe: int = 16,
) -> faiss.Index:
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index


def index_num_bytes(index: faiss.Index) -> int:
    return len(faiss.serialize_index(index))


def search(index: faiss.Index, queries: np.ndarray, k: int) -> np.ndarray:
    _scores, ids = index.search(np.ascontiguousarray(queries, dtype="float32"), k)
    return ids


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
