from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm import LLM
from app.logging_conf import get_logger, new_correlation_id
from app.retriever import Retriever
from app.router import (
    CALCULATE,
    DIRECT_ANSWER,
    RETRIEVE_AND_ANSWER,
    RouterResult,
    classify_intent,
    handle_calculate,
    handle_direct,
    handle_retrieve,
)

log = get_logger("app.router")


class _State(TypedDict, total=False):
    query: str
    intent: str
    result: RouterResult


class LangGraphRouter:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self._retriever = retriever
        self._llm = llm
        self._graph = self._build()

    def _build(self):
        graph = StateGraph(_State)
        graph.add_node("classify", lambda s: {"intent": classify_intent(s["query"])})
        graph.add_node("calculate", lambda s: {"result": handle_calculate(s["query"])})
        graph.add_node(
            "retrieve",
            lambda s: {"result": handle_retrieve(s["query"], self._retriever, self._llm)},
        )
        graph.add_node(
            "direct",
            lambda s: {"result": handle_direct(s["query"], self._llm)},
        )

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            lambda s: s["intent"],
            {
                CALCULATE: "calculate",
                RETRIEVE_AND_ANSWER: "retrieve",
                DIRECT_ANSWER: "direct",
            },
        )
        for node in ("calculate", "retrieve", "direct"):
            graph.add_edge(node, END)
        return graph.compile()

    def route(self, query: str) -> RouterResult:
        cid = new_correlation_id()
        t0 = time.perf_counter()
        out = self._graph.invoke({"query": query})
        result: RouterResult = out["result"]
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
                "backend": "langgraph",
            },
        )
        return result
