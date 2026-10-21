from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import config
from app.llm import LLM, get_llm
from app.logging_conf import get_logger, new_correlation_id
from app.rag import answer_question
from app.retriever import Retriever, get_retriever
from app.tools import calculate, extract_expression

log = get_logger("app.router")

CALCULATE = "calculate"
RETRIEVE_AND_ANSWER = "retrieve_and_answer"
DIRECT_ANSWER = "direct_answer"

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|hiya)( there| everyone| all| folks)?[\s,.!?]*$"
    r"|^(thanks|thank you|thx|good (morning|afternoon|evening)|"
    r"how are you|who are you|what can you do)[\s,.!?]*$"
)

DIRECT_SYSTEM_PROMPT = (
    "You are a concise, friendly assistant for a retrieval system. Answer in one "
    "or two short sentences."
)


def classify_intent(query: str) -> str:
    q = query.strip().lower()
    if not q:
        return DIRECT_ANSWER
    if _GREETING_RE.match(q):
        return DIRECT_ANSWER
    if extract_expression(query) is not None:
        return CALCULATE
    return RETRIEVE_AND_ANSWER


@dataclass
class RouterResult:
    query: str
    intent: str
    answer: str
    tool: str
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    detail: dict = field(default_factory=dict)

    def to_response(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 2),
        }


def handle_calculate(query: str) -> RouterResult:
    expression = extract_expression(query)
    value = calculate(expression)
    rendered = int(value) if isinstance(value, float) and value.is_integer() else value
    answer = f"{expression} = {rendered}"
    return RouterResult(
        query=query, intent=CALCULATE, answer=answer, tool="calculator",
        detail={"expression": expression, "result": value},
    )


def handle_direct(query: str, llm: LLM | None = None) -> RouterResult:
    llm = llm or get_llm()
    gen = llm.generate(
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return RouterResult(
        query=query, intent=DIRECT_ANSWER, answer=gen.text, tool="llm",
        detail={
            "completion_tokens": gen.completion_tokens,
            "tokens_per_second": round(gen.tokens_per_second, 2),
        },
    )


def handle_retrieve(
    query: str,
    retriever: Retriever | None = None,
    llm: LLM | None = None,
    top_k: int | None = None,
) -> RouterResult:
    result = answer_question(query, retriever=retriever, llm=llm, top_k=top_k)
    return RouterResult(
        query=query, intent=RETRIEVE_AND_ANSWER, answer=result.answer,
        tool="retriever+llm", sources=result.sources,
        detail={
            "completion_tokens": result.completion_tokens,
            "tokens_per_second": round(result.tokens_per_second, 2),
        },
    )


class Router:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self._retriever = retriever
        self._llm = llm

    def route(self, query: str) -> RouterResult:
        cid = new_correlation_id()
        t0 = time.perf_counter()
        intent = classify_intent(query)

        if intent == CALCULATE:
            result = handle_calculate(query)
        elif intent == DIRECT_ANSWER:
            result = handle_direct(query, self._llm)
        else:
            result = handle_retrieve(query, self._retriever, self._llm)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "routing decision",
            extra={
                "correlation_id": cid,
                "intent": result.intent,
                "tool": result.tool,
                "latency_ms": round(result.latency_ms, 2),
                "n_sources": len(result.sources),
                "query_preview": query[:80],
                "backend": "plain",
            },
        )
        return result


def route_query(query: str, *, backend: str | None = None) -> RouterResult:
    backend = backend or config.router_backend
    if backend == "langgraph":
        from app.router_langgraph import LangGraphRouter

        return LangGraphRouter().route(query)
    return Router().route(query)
