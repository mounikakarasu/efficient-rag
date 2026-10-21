import csv

from bench.benchmark import _pct, energy_measurement_mode, write_csv, write_markdown


def test_percentile_helper():
    values = [10, 20, 30, 40, 50]
    assert _pct(values, 50) == 30
    assert _pct([], 95) == 0.0


def test_energy_mode_is_labeled():
    mode = energy_measurement_mode()
    assert "measured" in mode or "estimated" in mode


def _fake_report():
    return {
        "summary": {
            "questions": 2, "index_build_seconds": 1.2, "documents_indexed": 1,
            "chunks_indexed": 3, "latency_ms_p50": 100.0, "latency_ms_p95": 200.0,
            "rag_latency_ms_p50": 150.0, "rag_latency_ms_p95": 220.0,
            "tokens_per_second_median": 5.5, "peak_rss_gb": 2.3,
            "energy_kwh": 0.0012, "co2_kg": 0.00005,
            "energy_mode": "estimated (CodeCarbon TDP model)", "wall_seconds": 10.0,
        },
        "rows": [
            {"id": 1, "category": "retrieve_and_answer", "intent": "retrieve_and_answer",
             "latency_ms": 150.0, "completion_tokens": 12, "tokens_per_second": 5.5},
            {"id": 2, "category": "calculate", "intent": "calculate",
             "latency_ms": 0.5, "completion_tokens": 0, "tokens_per_second": 0.0},
        ],
        "system": {
            "cpu": "Test CPU", "physical_cores": 4, "logical_cores": 8,
            "ram_total_gb": 16.0, "os": "Windows 11 (build 26200)", "python": "3.11.6",
            "torch": "2.12.1+cpu", "gpu_used": False, "cuda_available": False,
        },
        "config": {
            "embedding_model": "all-MiniLM-L6-v2", "llm_model": "qwen.gguf",
            "chunk_size": 800, "chunk_overlap": 120, "top_k": 4,
            "router_backend": "plain", "n_gpu_layers": 0,
        },
        "timestamp": "2026-06-22T00:00:00+00:00",
    }


def test_write_markdown(tmp_path):
    out = tmp_path / "benchmarks.md"
    write_markdown(_fake_report(), out)
    text = out.read_text(encoding="utf-8")
    assert "# Benchmarks" in text
    assert "Test CPU" in text
    assert "Peak process RAM" in text
    assert "estimated" in text


def test_write_csv(tmp_path):
    out = tmp_path / "benchmarks.csv"
    write_csv(_fake_report(), out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["intent"] == "retrieve_and_answer"
    assert rows[1]["category"] == "calculate"
