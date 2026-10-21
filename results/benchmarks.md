# Benchmarks

_Generated 2026-06-22T05:47:53+00:00 by `python -m bench.benchmark` on the hardware listed; no values are hand-entered._

## Hardware & software

| Field | Value |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-1255U |
| Cores | 10 physical / 12 logical |
| RAM | 16.8 GB |
| Compute | CPU-only (no GPU offload) |
| OS | Windows 11 (build 26200) |
| Python | 3.11.6 |
| PyTorch | 2.12.1+cpu |

## Configuration

| Field | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | qwen2.5-1.5b-instruct-q4_k_m.gguf |
| Chunk size / overlap | 800 / 120 chars |
| top_k | 4 |
| Router backend | plain |
| GPU layers offloaded | 0 |

## Results

| Metric | Value |
|---|---|
| Questions evaluated | 12 |
| Index build time | 12.286 s (58 chunks from 1 docs) |
| Latency p50 (all) | 45876.3 ms |
| Latency p95 (all) | 64545.5 ms |
| RAG latency p50 | 51613.2 ms |
| RAG latency p95 | 65401.3 ms |
| Generation throughput (median) | 0.91 tokens/s |
| Peak process RAM | 2.66 GB |
| Energy (estimated (CodeCarbon TDP model)) | 0.004379 kWh |
| CO2 (estimated (CodeCarbon TDP model)) | 0.000178 kg |

> Energy/CO2 are estimated by CodeCarbon's TDP model (no Intel RAPL on this OS); latency, throughput, and memory are directly measured.

## Per-question detail

| id | category | intent | latency (ms) | gen tokens | tokens/s |
|---|---|---|---|---|---|
| 1 | retrieve_and_answer | retrieve_and_answer | 45222.49 | 53 | 1.17 |
| 2 | retrieve_and_answer | retrieve_and_answer | 46530.11 | 21 | 0.45 |
| 3 | retrieve_and_answer | retrieve_and_answer | 56566.22 | 26 | 0.46 |
| 4 | retrieve_and_answer | retrieve_and_answer | 53954.2 | 50 | 0.93 |
| 5 | retrieve_and_answer | retrieve_and_answer | 51613.18 | 110 | 2.14 |
| 6 | retrieve_and_answer | retrieve_and_answer | 43820.13 | 20 | 0.46 |
| 7 | retrieve_and_answer | retrieve_and_answer | 61978.03 | 111 | 1.8 |
| 8 | retrieve_and_answer | retrieve_and_answer | 37348.22 | 33 | 0.89 |
| 9 | retrieve_and_answer | retrieve_and_answer | 67683.42 | 17 | 0.25 |
| 10 | calculate | calculate | 2.2 | 0 | 0.0 |
| 11 | calculate | calculate | 0.7 | 0 | 0.0 |
| 12 | direct_answer | direct_answer | 3744.04 | 9 | 2.4 |
