# Efficient RAG — Design Notes

## What the system does

Efficient RAG is an on-device Retrieval-Augmented Generation assistant. It pairs
a small quantized language model with a FAISS vector index so that questions are
answered using retrieved source passages rather than the model's memory alone.

## Why efficiency matters

The project extends the efficiency lens of the IEEE paper "Lean Fine-Tuning on a
Budget" from training to inference. The goal is to run within a tight memory
budget — a target of roughly four to six gigabytes of peak memory — on a CPU,
without requiring a datacenter GPU.

## Retrieval

Documents are split into overlapping chunks. Each chunk is embedded with the
all-MiniLM-L6-v2 sentence-transformer, which produces 384-dimensional vectors.
The vectors are stored in a FAISS inner-product index. Because the embeddings are
L2-normalized, inner-product search is equivalent to cosine similarity.

## Generation

For each query the system retrieves the top-k most similar chunks, builds a
grounded prompt that includes those passages as context, and asks the quantized
model to answer using only that context. The response includes the answer, the
source chunks it was grounded in, and the measured latency in milliseconds.

## The agentic router

A lightweight router classifies each query before answering. Arithmetic questions
are sent to a calculator tool, knowledge questions trigger retrieval, and simple
conversational messages are answered directly. Every routing decision is logged
as structured JSON with a correlation ID so the system's behavior is auditable.
