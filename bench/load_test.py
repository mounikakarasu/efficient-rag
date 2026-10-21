from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

QUERIES = [
    "How do I read a file line by line in Python?",
    "How to reverse a string in Java?",
    "What is the difference between a list and a tuple in Python?",
    "How do I write an inner join in SQL?",
    "How to center a div with CSS flexbox?",
    "What is 128 * 47?",
    "How to catch an exception in Python?",
    "How do I sort a dictionary by value?",
]


def _one(client: httpx.Client, url: str, question: str) -> tuple[float, int, bool]:
    t0 = time.perf_counter()
    try:
        r = client.post(url, json={"question": question}, timeout=120.0)
        latency = (time.perf_counter() - t0) * 1000
        cached = bool(r.json().get("cached", False)) if r.status_code == 200 else False
        return latency, r.status_code, cached
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0, False


def run(url: str, requests: int, concurrency: int, warm: bool) -> dict:
    with httpx.Client() as client:
        if warm:
            for q in QUERIES:
                _one(client, url, q)

        work = [random.choice(QUERIES) for _ in range(requests)]
        latencies, statuses, cached_flags = [], [], []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for latency, status, cached in pool.map(lambda q: _one(client, url, q), work):
                latencies.append(latency)
                statuses.append(status)
                cached_flags.append(cached)
        wall = time.perf_counter() - t0

    ok = sum(1 for s in statuses if s == 200)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "ok": ok,
        "errors": requests - ok,
        "throughput_rps": round(requests / wall, 1) if wall else 0.0,
        "cache_hit_rate": round(sum(cached_flags) / requests, 3) if requests else 0.0,
        "latency_ms_p50": round(float(np.percentile(latencies, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
        "latency_ms_p99": round(float(np.percentile(latencies, 99)), 1),
        "wall_seconds": round(wall, 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load-test the /ask endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-warm", action="store_true")
    args = parser.parse_args(argv)

    stats = run(args.url, args.requests, args.concurrency, warm=not args.no_warm)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
