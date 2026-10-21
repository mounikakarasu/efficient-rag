import math

import numpy as np

from app.eval_metrics import (
    cosine,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l_f1,
    token_f1,
)


def test_recall_at_k():
    assert recall_at_k([3, 1, 2], 1, 3) == 1.0
    assert recall_at_k([3, 1, 2], 9, 3) == 0.0
    assert recall_at_k([3, 1, 2], 2, 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 1, 2], 1) == 0.5
    assert reciprocal_rank([1, 2], 1) == 1.0
    assert reciprocal_rank([2, 3], 1) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k([1, 2, 3], 1, 3) == 1.0
    assert abs(ndcg_at_k([2, 1, 3], 1, 3) - 1 / math.log2(3)) < 1e-9
    assert ndcg_at_k([2, 3], 1, 2) == 0.0


def test_token_f1():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("", "x") == 0.0
    assert 0 < token_f1("the cat", "the cat sat") < 1


def test_rouge_l_f1():
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert rouge_l_f1("", "x") == 0.0
    assert 0 < rouge_l_f1("a b c d", "a c d") <= 1


def test_cosine():
    a = np.array([1, 0, 0], dtype="float32")
    assert abs(cosine(a, a) - 1.0) < 1e-6
    assert abs(cosine(a, np.array([0, 1, 0], dtype="float32"))) < 1e-6
