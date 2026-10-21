## Who I am / why this project exists

I'm an applied ML/LLM engineer. I'm first author on an IEEE paper, **"Lean
Fine-Tuning on a Budget,"** which studied parameter-efficient LLM adaptation
(LoRA, (IA)³, knowledge distillation) under a fixed 16 GB consumer-GPU budget,
measuring accuracy, memory, latency, and CO₂. This project extends that same
efficiency lens from **training** to **RAG inference**. That framing must come
through in the README and the metrics.

## Goal

A working, benchmarked Retrieval-Augmented Generation assistant that:

- Runs a quantized small LLM + FAISS retrieval within a tight memory budget on
  CPU or a single consumer GPU.
- Includes a lightweight agentic router (tool-use / function-calling) that
  decides how to handle each query.
- Is instrumented for latency, memory, and energy so I can report real
  efficiency numbers.

## Hard constraints / principles

- Must run on **CPU only** (and optionally use a single consumer GPU if present).
  Target peak memory under **~4–6 GB**.
- **Reproducible:** pin dependencies in `requirements.txt`, set seeds, and
  document the exact hardware used for any reported number.
- **Sustainability:** log energy + estimated CO₂ with CodeCarbon, mirroring my
  paper's methodology.
- **Shippable in one focused session:** build incrementally and commit after
  every phase with a clear message. Prefer a small, correct, well-documented
  system over a sprawling one.
- Keep dependencies lean; no heavyweight orchestration unless it earns its place.

## Tech stack (use unless you hit a real blocker — then tell me)

- Python 3.11
- Embeddings: sentence-transformers → `all-MiniLM-L6-v2` (small, CPU-friendly)
- Vector store: `faiss-cpu`
- LLM: 4-bit GGUF via `llama-cpp-python`. Default model:
  **Qwen2.5-1.5B-Instruct (Q4_K_M)** (~1 GB). Make the model path/name a config
  value so I can swap to Phi-3-mini or similar.
- API: FastAPI + Uvicorn
- Instrumentation: `codecarbon`, `time`, `psutil` (RAM), and
  `torch.cuda.max_memory_allocated()` if a GPU is used
- Packaging: Docker + docker-compose
- Tests: `pytest` for retrieval and router logic
- Optional: a thin **LangGraph** version of the router in addition to the
  plain-Python one, so I can legitimately claim LangGraph familiarity — but only
  if it stays readable. 

## Architecture

1. **Ingest pipeline** — load documents (PDF / markdown / txt) from `./data`,
   chunk them (configurable size + overlap), embed, and persist a FAISS index +
   metadata to `./index`.
2. **Query pipeline** — embed the query → top-k retrieve → build a grounded
   prompt with sources → quantized LLM generates → return
   `{answer, sources, latency_ms}`.
3. **Agentic router** — classify each query into one of: `retrieve_and_answer`,
   `calculate` (route to a calculator tool), or `direct_answer`. Dispatch the
   right tool, optionally do a second pass to compose the final answer. Log every
   routing decision as structured JSON with a correlation ID.
4. **Benchmark harness** — measure median + p95 latency, tokens/sec, index build
   time, peak RAM (and VRAM if GPU), and CodeCarbon energy/CO₂ over a fixed
   question set. Write results to `results/benchmarks.md` (table) and
   `results/benchmarks.csv`.

