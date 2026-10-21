from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.config import config
from app.logging_conf import get_logger

log = get_logger("app.access")

TENANTS = ("team-alpha", "team-beta", "team-gamma")


def doc_tenant(doc_id: str) -> str:
    digest = int(hashlib.sha1(str(doc_id).encode("utf-8")).hexdigest(), 16)
    return TENANTS[digest % len(TENANTS)]


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset

    def can_access(self, doc_id: str) -> bool:
        return doc_tenant(doc_id) in self.groups


class SecureRetriever:
    def __init__(self, base, principal: Principal, overfetch: int = 6):
        self._base = base
        self._principal = principal
        self._overfetch = overfetch

    def search(self, query: str, top_k: int | None = None):
        top_k = top_k or config.top_k
        candidates = self._base.search(query, top_k * self._overfetch)
        allowed = [c for c in candidates if self._principal.can_access(c.doc_id)]
        denied = len(candidates) - len(allowed)
        if denied:
            log.info(
                "access filter applied",
                extra={"user": self._principal.user_id, "denied": denied, "kept": len(allowed)},
            )
        return allowed[:top_k]
