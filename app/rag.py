from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from app.config import config
from app.llm import LLM, get_llm
from app.logging_conf import get_logger
from app.retriever import RetrievedChunk, Retriever, get_retriever

log = get_logger("app.rag")

SYSTEM_PROMPT = (
    "You are a precise retrieval assistant. Answer the user's question using "
    "ONLY the provided context passages. If the answer is not contained in the "
    "context, say you don't have enough information rather than guessing. Be "
    "concise and factual, and do not invent sources."
)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    if chunks:
        context = "\n\n".join(
            f"[Source {i + 1}: {c.source}]\n{c.text}" for i, c in enumerate(chunks)
        )
    else:
        context = "(no relevant context found)"
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0

    def to_response(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 2),
        }


def answer_question(
    question: str,
    *,
    retriever: Retriever | None = None,
    llm: LLM | None = None,
    top_k: int | None = None,
) -> RagAnswer:
    retriever = retriever or get_retriever()
    llm = llm or get_llm()

    t0 = time.perf_counter()
    chunks = retriever.search(question, top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    messages = build_messages(question, chunks)
    gen = llm.generate(messages=messages)

    total_ms = retrieval_ms + gen.latency_ms
    log.info(
        "rag answer",
        extra={
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen.latency_ms, 2),
            "latency_ms": round(total_ms, 2),
            "n_sources": len(chunks),
            "completion_tokens": gen.completion_tokens,
            "tokens_per_second": round(gen.tokens_per_second, 2),
        },
    )
    return RagAnswer(
        answer=gen.text,
        sources=[c.as_source() for c in chunks],
        latency_ms=total_ms,
        retrieval_ms=retrieval_ms,
        generation_ms=gen.latency_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        tokens_per_second=gen.tokens_per_second,
    )


def answer_question_stream(
    question: str,
    *,
    retriever: Retriever | None = None,
    llm: LLM | None = None,
    top_k: int | None = None,
) -> tuple[list[dict], Iterator[str]]:
    retriever = retriever or get_retriever()
    llm = llm or get_llm()
    chunks = retriever.search(question, top_k)
    messages = build_messages(question, chunks)
    sources = [c.as_source() for c in chunks]
    return sources, llm.generate_stream(messages=messages)
