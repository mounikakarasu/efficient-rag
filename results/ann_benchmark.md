# ANN Benchmark

_150,093 vectors (384-dim) over 200 real Stack Overflow queries; recall@10 measured against exact (Flat) search._

| Index | Params | recall@10 | p50 (ms) | p95 (ms) | Index size (MB) | Build (s) |
|---|---|---|---|---|---|---|
| Flat (exact) | brute force | 1.0 | 13.365 | 15.999 | 230.5 | 0.15 |
| HNSW | M=32, efSearch=64 | 0.988 | 0.354 | 0.472 | 271.4 | 43.49 |
| IVF-PQ | nlist=2048, m=48, nprobe=16 | 0.646 | 0.272 | 1.834 | 12.0 | 92.16 |

Flat (exact) index is 230.5 MB with p50 13.365 ms — the baseline ANN is compared against.

## IVF-PQ nprobe sweep (recall vs latency trade-off)

| nprobe | recall@10 | p50 (ms) |
|---|---|---|
| 4 | 0.574 | 0.205 |
| 8 | 0.623 | 0.268 |
| 16 | 0.646 | 0.313 |
| 32 | 0.655 | 0.33 |
| 64 | 0.662 | 0.581 |
