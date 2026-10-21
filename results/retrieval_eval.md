# Retrieval Quality

_Recall@10 against 200 labeled Stack Overflow queries (relevant doc = the question the query was drawn from)._

| Method | recall@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|
| Dense (HNSW) | 0.8 | 5.9 | 8.4 |
| Hybrid (BM25+dense+RRF) | 0.98 | 1030.5 | 1247.4 |
| Hybrid + cross-encoder rerank | 1.0 | 1828.8 | 3710.8 |
