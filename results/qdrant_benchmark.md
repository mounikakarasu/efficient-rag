# FAISS vs Qdrant

_150,093 vectors, 200 real queries, recall@10 vs exact search. Both use HNSW (m=32)._

| Backend | recall@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|
| FAISS HNSW (in-process) | 0.9875 | 0.378 | 1.288 |
| Qdrant HNSW (service) | 0.999 | 18.023 | 35.731 |

## Metadata filtering (Qdrant only)

A filtered search for doc_id `81495` returned 4 results, all from that document: passed. FAISS has no native payload filtering; Qdrant applies it inside the search.

FAISS runs in-process (no network hop); Qdrant runs as a service and adds persistence, concurrency, payload filtering, and built-in quantization at the cost of a localhost round-trip.
