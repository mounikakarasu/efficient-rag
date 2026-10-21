from __future__ import annotations

import argparse
import csv
import json
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from app.config import config
from app.ingest import build_index
from app.logging_conf import configure_logging, get_logger
from app.llm import get_llm
from app.retriever import Retriever
from app.router import Router

log = get_logger("bench")

ROOT = Path(__file__).resolve().parent.parent


def _cpu_name() -> str:
    try:
        from cpuinfo import get_cpu_info

        return get_cpu_info().get("brand_raw") or platform.processor()
    except Exception:
        return platform.processor() or "unknown"


def _os_string() -> str:
    if platform.system() == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
            name = "11" if build >= 22000 else "10"
            return f"Windows {name} (build {build})"
        except Exception:
            return f"Windows {platform.release()}"
    return f"{platform.system()} {platform.release()}"


def system_info() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": _os_string(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_used": bool(cuda and config.llm_n_gpu_layers > 0),
        "cuda_available": cuda,
    }


def energy_measurement_mode() -> str:
    if Path("/sys/class/powercap/intel-rapl").exists():
        return "measured (Intel RAPL)"
    return "estimated (CodeCarbon TDP model)"


class PeakMemorySampler(threading.Thread):
    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self._proc = psutil.Process()
        self._interval = interval
        self._stop_event = threading.Event()
        self.peak_rss = self._proc.memory_info().rss

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            self._stop_event.wait(self._interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak_rss


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def run_benchmark(questions_path: Path, corpus_dir: Path) -> dict:
    configure_logging("WARNING")

    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload["questions"]

    build_stats = build_index(corpus_dir, config.index_dir, config)
    index_build_seconds = build_stats["index_build_seconds"]

    retriever = Retriever.load()
    llm = get_llm()
    router = Router(retriever=retriever, llm=llm)
    router.route("Warm-up query about the solar system.")

    from codecarbon import EmissionsTracker

    sampler = PeakMemorySampler()
    sampler.start()
    tracker = EmissionsTracker(save_to_file=False, log_level="error", tracking_mode="process")
    tracker.start()

    rows: list[dict] = []
    wall_t0 = time.perf_counter()
    for q in questions:
        result = router.route(q["question"])
        rows.append(
            {
                "id": q["id"],
                "category": q.get("category", ""),
                "intent": result.intent,
                "latency_ms": round(result.latency_ms, 2),
                "completion_tokens": result.detail.get("completion_tokens", 0),
                "tokens_per_second": result.detail.get("tokens_per_second", 0.0),
            }
        )
    wall_seconds = time.perf_counter() - wall_t0

    emissions_kg = tracker.stop() or 0.0
    energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", 0.0) or 0.0)
    peak_rss_gb = sampler.stop() / 1e9

    all_latency = [r["latency_ms"] for r in rows]
    rag_latency = [r["latency_ms"] for r in rows if r["intent"] == "retrieve_and_answer"]
    gen_tps = [r["tokens_per_second"] for r in rows if r["completion_tokens"] > 0]

    summary = {
        "questions": len(rows),
        "index_build_seconds": round(index_build_seconds, 3),
        "documents_indexed": build_stats["documents"],
        "chunks_indexed": build_stats["chunks"],
        "latency_ms_p50": round(_pct(all_latency, 50), 1),
        "latency_ms_p95": round(_pct(all_latency, 95), 1),
        "rag_latency_ms_p50": round(_pct(rag_latency, 50), 1),
        "rag_latency_ms_p95": round(_pct(rag_latency, 95), 1),
        "tokens_per_second_median": round(_pct(gen_tps, 50), 2),
        "peak_rss_gb": round(peak_rss_gb, 2),
        "energy_kwh": energy_kwh,
        "co2_kg": emissions_kg,
        "energy_mode": energy_measurement_mode(),
        "wall_seconds": round(wall_seconds, 2),
    }
    return {
        "summary": summary,
        "rows": rows,
        "system": system_info(),
        "config": {
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_filename,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "router_backend": config.router_backend,
            "n_gpu_layers": config.llm_n_gpu_layers,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "category", "intent", "latency_ms",
                        "completion_tokens", "tokens_per_second"],
        )
        writer.writeheader()
        writer.writerows(report["rows"])


def write_markdown(report: dict, path: Path) -> None:
    s, sys_, cfg = report["summary"], report["system"], report["config"]
    gpu_line = (
        f"{sys_['cpu']} GPU offload"
        if sys_["gpu_used"]
        else "CPU-only (no GPU offload)"
    )
    energy_note = (
        "Energy/CO2 are estimated by CodeCarbon's TDP model (no Intel RAPL on "
        "this OS); latency, throughput, and memory are directly measured."
        if "estimated" in s["energy_mode"]
        else "Energy/CO2 measured via Intel RAPL."
    )

    lines = [
        "# Benchmarks",
        "",
        f"_Generated {report['timestamp']} by `python -m bench.benchmark` on the "
        "hardware listed; no values are hand-entered._",
        "",
        "## Hardware & software",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| CPU | {sys_['cpu']} |",
        f"| Cores | {sys_['physical_cores']} physical / {sys_['logical_cores']} logical |",
        f"| RAM | {sys_['ram_total_gb']} GB |",
        f"| Compute | {gpu_line} |",
        f"| OS | {sys_['os']} |",
        f"| Python | {sys_['python']} |",
        f"| PyTorch | {sys_['torch']} |",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Embedding model | {cfg['embedding_model']} |",
        f"| LLM | {cfg['llm_model']} |",
        f"| Chunk size / overlap | {cfg['chunk_size']} / {cfg['chunk_overlap']} chars |",
        f"| top_k | {cfg['top_k']} |",
        f"| Router backend | {cfg['router_backend']} |",
        f"| GPU layers offloaded | {cfg['n_gpu_layers']} |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions evaluated | {s['questions']} |",
        f"| Index build time | {s['index_build_seconds']} s "
        f"({s['chunks_indexed']} chunks from {s['documents_indexed']} docs) |",
        f"| Latency p50 (all) | {s['latency_ms_p50']} ms |",
        f"| Latency p95 (all) | {s['latency_ms_p95']} ms |",
        f"| RAG latency p50 | {s['rag_latency_ms_p50']} ms |",
        f"| RAG latency p95 | {s['rag_latency_ms_p95']} ms |",
        f"| Generation throughput (median) | {s['tokens_per_second_median']} tokens/s |",
        f"| Peak process RAM | {s['peak_rss_gb']} GB |",
        f"| Energy ({s['energy_mode']}) | {s['energy_kwh']:.6f} kWh |",
        f"| CO2 ({s['energy_mode']}) | {s['co2_kg']:.6f} kg |",
        "",
        f"> {energy_note}",
        "",
        "## Per-question detail",
        "",
        "| id | category | intent | latency (ms) | gen tokens | tokens/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['intent']} | {r['latency_ms']} "
            f"| {r['completion_tokens']} | {r['tokens_per_second']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Efficient RAG benchmark.")
    parser.add_argument("--questions", type=Path, default=ROOT / "bench" / "questions.json")
    parser.add_argument("--corpus", type=Path, default=config.data_dir)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    report = run_benchmark(args.questions, args.corpus)
    write_csv(report, args.out / "benchmarks.csv")
    write_markdown(report, args.out / "benchmarks.md")

    s = report["summary"]
    print(
        f"Benchmark complete: {s['questions']} questions | "
        f"RAG p50 {s['rag_latency_ms_p50']} ms / p95 {s['rag_latency_ms_p95']} ms | "
        f"{s['tokens_per_second_median']} tok/s | peak RAM {s['peak_rss_gb']} GB | "
        f"{s['energy_kwh']:.6f} kWh ({s['energy_mode']})"
    )
    print(f"Wrote {args.out / 'benchmarks.md'} and {args.out / 'benchmarks.csv'}")


if __name__ == "__main__":
    main()
