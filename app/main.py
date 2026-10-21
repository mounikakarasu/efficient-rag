from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import __version__
from app.access import Principal, SecureRetriever
from app.auth import AuthError, principal_from_token
from app.cache import get_cache
from app.config import config
from app.logging_conf import configure_logging, get_logger
from app.rag import answer_question_stream
from app.retriever import get_retriever
from app.router import Router, route_query

configure_logging(config.log_level)
log = get_logger("app.main")

_STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("service starting", extra={"version": __version__})
    yield
    log.info("service stopping")


app = FastAPI(
    title="Efficient RAG",
    version=__version__,
    description="On-device Retrieval-Augmented Generation assistant (CPU-first, benchmarked).",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    intent: str
    sources: list[Source] = []
    latency_ms: float
    cached: bool = False


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "embedding_model": config.embedding_model,
        "llm_model": config.llm_filename,
        "router_backend": config.router_backend,
    }


def _principal_from_headers(user_id: str | None, groups: str | None) -> Principal | None:
    if not groups:
        return None
    parsed = frozenset(g.strip() for g in groups.split(",") if g.strip())
    if not parsed:
        return None
    return Principal(user_id=user_id or "anonymous", groups=parsed)


def _resolve_principal(authorization, x_user_id, x_user_groups) -> Principal | None:
    if authorization and authorization.lower().startswith("bearer "):
        return principal_from_token(authorization[7:].strip())
    if config.auth_required:
        raise AuthError("authentication required")
    return _principal_from_headers(x_user_id, x_user_groups)


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_user_groups: str | None = Header(default=None),
) -> dict:
    try:
        principal = _resolve_principal(authorization, x_user_id, x_user_groups)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    cache = get_cache() if (config.cache_enabled and principal is None) else None
    query_vec = None
    if cache is not None:
        cached, query_vec = cache.lookup(req.question)
        if cached is not None:
            return {**cached, "cached": True}

    try:
        if principal is not None:
            secured = SecureRetriever(get_retriever(), principal)
            result = Router(retriever=secured).route(req.question)
        else:
            result = route_query(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = result.to_response()
    if cache is not None and query_vec is not None:
        cache.store(query_vec, response)
    return {**response, "cached": False}


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    try:
        sources, token_stream = answer_question_stream(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def events():
        started = time.time()
        yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"
        for piece in token_stream:
            yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"
        yield f"event: done\ndata: {json.dumps({'latency_ms': round((time.time() - started) * 1000, 2)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
