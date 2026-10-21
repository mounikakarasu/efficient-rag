from __future__ import annotations

from pathlib import Path

from app.config import config
from app.embeddings import embed_texts
from app.logging_conf import get_logger
from app.rerank import rerank as rerank_fn
from app.retriever import RetrievedChunk
from app.sparse import BM25Index
from app.vector_index import load_chunk_meta, load_index
from app.vector_index import search as faiss_search

log = get_logger("app.hybrid")


def rrf_fuse(rankings: list[list[int]], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class HybridRetriever:
    def __init__(self, index, bm25: BM25Index, metadata: list[dict], rerank_enabled: bool | None = None):
        self._index = index
        self._bm25 = bm25
        self._metadata = metadata
        self._rerank_enabled = config.rerank_enabled if rerank_enabled is None else rerank_enabled

    @classmethod
    def load(cls, embeddings_dir: Path | None = None, rerank_enabled: bool | None = None) -> "HybridRetriever":
        embeddings_dir = embeddings_dir or config.embeddings_dir
        index = load_index(embeddings_dir / "ann.index")
        bm25 = BM25Index.load(embeddings_dir / "bm25.pkl")
        metadata = load_chunk_meta(embeddings_dir)
        if index.ntotal != len(metadata) or bm25.size != len(metadata):
            raise ValueError("Index/BM25/metadata size mismatch.")
        log.info("hybrid retriever loaded", extra={"vectors": index.ntotal, "rerank": rerank_enabled})
        return cls(index, bm25, metadata, rerank_enabled)

    def _dense_ids(self, query: str, candidate_k: int) -> list[int]:
        query_vec = embed_texts([query])
        ids = faiss_search(self._index, query_vec, min(candidate_k, len(self._metadata)))
        return [int(i) for i in ids[0] if i >= 0]

    def _sparse_ids(self, query: str, candidate_k: int) -> list[int]:
        return [i for i, _score in self._bm25.search(query, candidate_k)]

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or config.top_k
        if not query.strip() or not self._metadata:
            return []

        dense = self._dense_ids(query, config.hybrid_candidate_k)
        sparse = self._sparse_ids(query, config.hybrid_candidate_k)
        fused = rrf_fuse([dense, sparse], config.rrf_k)

        if self._rerank_enabled and fused:
            candidates = [(idx, self._metadata[idx]["text"]) for idx in fused[: config.rerank_top_n]]
            chosen = rerank_fn(query, candidates, top_k)
        else:
            chosen = [(idx, 1.0 / (config.rrf_k + rank + 1)) for rank, idx in enumerate(fused[:top_k])]

        results: list[RetrievedChunk] = []
        for idx, score in chosen:
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    id=idx,
                    text=meta["text"],
                    source=meta.get("url") or meta.get("doc_id") or meta.get("source", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    score=float(score),
                    doc_id=str(meta.get("doc_id", "")),
                )
            )
        return results
