from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    VectorParams,
)

from app.config import config
from app.logging_conf import get_logger
from app.vector_index import load_chunk_meta, load_vectors

log = get_logger("app.qdrant_store")


def get_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or config.qdrant_url)


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
    )


def upload(client: QdrantClient, name: str, embeddings_dir: Path | None = None, batch_size: int = 1024) -> dict:
    vectors = load_vectors(embeddings_dir)
    meta = load_chunk_meta(embeddings_dir)
    dim = int(vectors.shape[1])
    n = int(vectors.shape[0])

    create_collection(client, name, dim)
    payload = [
        {
            "doc_id": m.get("doc_id"),
            "url": m.get("url"),
            "chunk_index": m.get("chunk_index"),
            "text": m.get("text"),
        }
        for m in meta
    ]
    client.upload_collection(
        collection_name=name,
        vectors=vectors,
        payload=payload,
        ids=list(range(n)),
        batch_size=batch_size,
    )
    log.info("qdrant upload complete", extra={"points": n, "dim": dim, "collection": name})
    return {"points": n, "dim": dim, "collection": name}


def search(client: QdrantClient, name: str, query_vec: np.ndarray, top_k: int, doc_id: str | None = None):
    query_filter = None
    if doc_id is not None:
        query_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    result = client.query_points(
        collection_name=name,
        query=np.asarray(query_vec, dtype="float32").ravel().tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return result.points
