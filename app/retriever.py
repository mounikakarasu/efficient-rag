from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass
from pathlib import Path

import faiss

from app.config import config
from app.embeddings import embed_texts
from app.logging_conf import configure_logging, get_logger

log = get_logger("app.retriever")


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source: str
    chunk_index: int
    score: float
    doc_id: str = ""

    def as_source(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "text": self.text,
        }


class Retriever:
    def __init__(self, index: faiss.Index, metadata: list[dict]):
        self._index = index
        self._metadata = metadata

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "Retriever":
        index_dir = index_dir or config.index_dir
        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Build one with "
                f"`python -m app.ingest --src ./data` first."
            )
        index = faiss.read_index(str(index_path))
        metadata = []
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    metadata.append(json.loads(line))
        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs "
                f"{len(metadata)} metadata rows in {index_dir}."
            )
        log.info("retriever loaded", extra={"vectors": index.ntotal})
        return cls(index, metadata)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or config.top_k
        if not query.strip() or not self._metadata:
            return []
        query_vec = embed_texts([query])
        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_vec, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[int(idx)]
            results.append(
                RetrievedChunk(
                    id=meta["id"],
                    text=meta["text"],
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    score=float(score),
                )
            )
        return results


class AnnRetriever:
    def __init__(self, index: faiss.Index, metadata: list[dict]):
        self._index = index
        self._metadata = metadata

    @classmethod
    def load(cls, embeddings_dir: Path | None = None) -> "AnnRetriever":
        from app.vector_index import load_chunk_meta, load_index

        embeddings_dir = embeddings_dir or config.embeddings_dir
        index_path = embeddings_dir / "ann.index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No ANN index in {embeddings_dir}. Build one with "
                f"`python -m app.build_index` (after `python -m app.embed_corpus`)."
            )
        index = load_index(index_path)
        metadata = load_chunk_meta(embeddings_dir)
        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs {len(metadata)} rows."
            )
        log.info("ann retriever loaded", extra={"vectors": index.ntotal})
        return cls(index, metadata)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or config.top_k
        if not query.strip() or not self._metadata:
            return []
        query_vec = embed_texts([query])
        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_vec, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[int(idx)]
            results.append(
                RetrievedChunk(
                    id=int(idx),
                    text=meta["text"],
                    source=meta.get("url") or meta.get("doc_id") or meta.get("source", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    score=float(score),
                    doc_id=str(meta.get("doc_id", "")),
                )
            )
        return results


class QdrantRetriever:
    def __init__(self, client, collection: str):
        self._client = client
        self._collection = collection

    @classmethod
    def load(cls, url: str | None = None, collection: str | None = None) -> "QdrantRetriever":
        from app.qdrant_store import get_client

        client = get_client(url)
        collection = collection or config.qdrant_collection
        client.get_collection(collection)
        log.info("qdrant retriever loaded", extra={"collection": collection})
        return cls(client, collection)

    def search(self, query: str, top_k: int | None = None, doc_id: str | None = None) -> list[RetrievedChunk]:
        from app.qdrant_store import search as qsearch

        top_k = top_k or config.top_k
        if not query.strip():
            return []
        query_vec = embed_texts([query])[0]
        points = qsearch(self._client, self._collection, query_vec, top_k, doc_id=doc_id)

        results: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    id=int(point.id),
                    text=payload.get("text", ""),
                    source=payload.get("url") or payload.get("doc_id") or "",
                    chunk_index=payload.get("chunk_index", 0),
                    score=float(point.score),
                    doc_id=str(payload.get("doc_id", "")),
                )
            )
        return results


@functools.lru_cache(maxsize=1)
def get_retriever():
    if config.vector_backend == "qdrant":
        return QdrantRetriever.load()
    emb = config.embeddings_dir
    if config.hybrid_enabled and (emb / "ann.index").exists() and (emb / "bm25.pkl").exists():
        from app.hybrid import HybridRetriever

        return HybridRetriever.load()
    if (emb / "ann.index").exists():
        return AnnRetriever.load()
    return Retriever.load()


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Query the FAISS index.")
    parser.add_argument("query")
    parser.add_argument("-k", "--top-k", type=int, default=config.top_k)
    args = parser.parse_args(argv)

    for rank, hit in enumerate(Retriever.load().search(args.query, args.top_k), 1):
        snippet = hit.text.replace("\n", " ")[:140]
        print(f"{rank}. [{hit.score:.3f}] {hit.source}#{hit.chunk_index}  {snippet}")


if __name__ == "__main__":
    main()
