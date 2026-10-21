from __future__ import annotations

import os
import socket
from contextlib import contextmanager

from app.config import config
from app.logging_conf import get_logger

log = get_logger("app.offline")


class EgressBlocked(Exception):
    pass


def _is_local(host: str) -> bool:
    h = str(host)
    return (
        h in ("localhost", "::1", "0.0.0.0")
        or h.startswith("127.")
        or h.startswith("::ffff:127.")
    )


@contextmanager
def no_external_egress():
    original = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if _is_local(host):
            return original(self, address)
        raise EgressBlocked(f"blocked outbound connection to {host}")

    socket.socket.connect = guarded
    try:
        yield
    finally:
        socket.socket.connect = original


def enforce_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    log.info("offline mode enforced")


def missing_models() -> list[str]:
    missing = []
    if not config.llm_model_path.exists():
        missing.append(str(config.llm_model_path))
    return missing
