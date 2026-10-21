# Load Test

_`python -m bench.load_test --requests 200 --concurrency 8` against a local Uvicorn
instance (i7-1255U), cache warmed with the query set, drawing from 8 repeated
Stack Overflow queries (steady-state popular-query traffic)._

| Metric | Value |
|---|---|
| Requests | 200 |
| Concurrency | 8 |
| Errors | 0 |
| Throughput | 75.8 req/s |
| Cache hit rate | 100% |
| Latency p50 | 100.4 ms |
| Latency p95 | 117.7 ms |
| Latency p99 | 237.7 ms |

## Containerized (2 vCPU / 4 GB limit, CPU only)

_`python -m bench.load_test --url http://localhost:8000/ask --requests 100 --concurrency 4`
against the Docker container under its resource limit._

| Metric | Value |
|---|---|
| Requests | 100 |
| Concurrency | 4 |
| Errors | 0 |
| Throughput | 31.1 req/s |
| Cache hit rate | 100% |
| Latency p50 | 112.2 ms |
| Latency p95 | 197.6 ms |
| Latency p99 | 420.2 ms |
| Container peak memory | 3.67 GB / 4 GB (92%) |

The ~100 ms latency floor is the query-embedding cost of the semantic cache lookup
(the cache still embeds each query to match semantically). Cold, uncached
retrieve+generate requests are LLM-generation-bound (seconds) and serialize on the
single quantized model instance; the cache is what makes popular-query serving fast.
The full hybrid stack (BM25 + dense HNSW + cross-encoder + quantized LLM) fits inside
the 2 vCPU / 4 GB budget at 3.67 GB peak.
