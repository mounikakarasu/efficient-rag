from app.cache import SemanticCache


def test_cache_hit_on_identical_query():
    cache = SemanticCache(threshold=0.95, max_size=10)
    value, qv = cache.lookup("how do I read a file in python")
    assert value is None
    cache.store(qv, {"answer": "use open()"})
    hit, _ = cache.lookup("how do I read a file in python")
    assert hit == {"answer": "use open()"}


def test_cache_miss_on_dissimilar_query():
    cache = SemanticCache(threshold=0.9, max_size=10)
    _, qv = cache.lookup("python file reading with readline")
    cache.store(qv, {"answer": "x"})
    hit, _ = cache.lookup("java string reversal and bubble sort algorithms")
    assert hit is None


def test_cache_evicts_oldest_beyond_max_size():
    cache = SemanticCache(threshold=0.99, max_size=2)
    for i, q in enumerate(["alpha topic one", "beta topic two", "gamma topic three"]):
        _, qv = cache.lookup(q)
        cache.store(qv, {"i": i})
    assert len(cache._entries) == 2
