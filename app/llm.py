from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.config import config
from app.logging_conf import get_logger
from app.model_download import ensure_model

log = get_logger("app.llm")


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tokens_per_second: float


class LLM:
    def __init__(self, llama):
        self._llama = llama

    @classmethod
    def load(cls, model_path: Path | None = None) -> "LLM":
        from llama_cpp import Llama

        model_path = model_path or ensure_model()
        log.info(
            "loading llm",
            extra={
                "model_path": str(model_path),
                "n_ctx": config.llm_n_ctx,
                "n_gpu_layers": config.llm_n_gpu_layers,
            },
        )
        llama = Llama(
            model_path=str(model_path),
            n_ctx=config.llm_n_ctx,
            n_threads=config.llm_n_threads or None,
            n_gpu_layers=config.llm_n_gpu_layers,
            verbose=False,
        )
        return cls(llama)

    def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Generation:
        if (messages is None) == (prompt is None):
            raise ValueError("Pass exactly one of `messages` or `prompt`.")
        max_tokens = max_tokens or config.llm_max_tokens
        temperature = config.llm_temperature if temperature is None else temperature

        t0 = time.perf_counter()
        if messages is not None:
            resp = self._llama.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stop=stop,
            )
            text = resp["choices"][0]["message"]["content"] or ""
        else:
            resp = self._llama.create_completion(
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, stop=stop,
            )
            text = resp["choices"][0]["text"] or ""
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = resp.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        tps = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
        return Generation(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            tokens_per_second=tps,
        )


    def generate_stream(
        self,
        *,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        max_tokens = max_tokens or config.llm_max_tokens
        temperature = config.llm_temperature if temperature is None else temperature
        stream = self._llama.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, stop=stop, stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0].get("delta", {}).get("content")
            if piece:
                yield piece


@functools.lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM.load()
