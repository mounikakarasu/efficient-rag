import numpy as np

from app.vector_index import build_flat, build_hnsw, build_ivfpq, index_num_bytes, search


def _rand_norm(n, seed, dim=384):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype("float32")
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True))


def test_flat_returns_exact_self():
    v = _rand_norm(200, 0)
    ids = search(build_flat(v), v[:5], 1)
    assert [int(ids[i][0]) for i in range(5)] == [0, 1, 2, 3, 4]


def test_hnsw_finds_self_top1():
    v = _rand_norm(2000, 1)
    hnsw = build_hnsw(v, m=32, ef_construction=200, ef_search=64)
    ids = search(hnsw, v[:50], 1)
    matches = sum(int(ids[i][0]) == i for i in range(50))
    assert matches >= 48


def test_ivfpq_recalls_self_in_topk():
    v = _rand_norm(5000, 2)
    ivfpq = build_ivfpq(v, nlist=64, m=48, nbits=8, nprobe=8)
    ids = search(ivfpq, v[:50], 10)
    assert ids.shape == (50, 10)
    found = sum(i in set(int(x) for x in ids[i]) for i in range(50))
    assert found >= 40


def test_index_num_bytes_positive():
    v = _rand_norm(100, 3)
    assert index_num_bytes(build_flat(v)) > 0
