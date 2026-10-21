from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def recall_at_k(retrieved_ids: list, relevant_id, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list, relevant_id) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_id, k: int) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == relevant_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def token_f1(pred: str, truth: str) -> float:
    p, t = Counter(_tokens(pred)), Counter(_tokens(truth))
    common = sum((p & t).values())
    if common == 0 or not p or not t:
        return 0.0
    precision = common / sum(p.values())
    recall = common / sum(t.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, truth: str) -> float:
    a, b = _tokens(pred), _tokens(truth)
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(a)
    recall = lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
