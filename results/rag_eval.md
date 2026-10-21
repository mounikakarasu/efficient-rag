# RAG Evaluation

_Offline, judge-free metrics (no external API). Retrieval scored against labeled relevant docs; answers scored against ground-truth accepted answers._

## Retrieval (hybrid + rerank, 100 queries)

| Metric | Value |
|---|---|
| recall@10 | 1.0 |
| MRR | 1.0 |
| nDCG@10 | 1.0 |

## Generation (20 queries)

| Metric | Value |
|---|---|
| Answer token-F1 (vs ground truth) | 0.2438 |
| Answer ROUGE-L (vs ground truth) | 0.1729 |
| Answer embedding similarity | 0.5003 |
| Faithfulness proxy (answer vs context) | 0.7143 |
