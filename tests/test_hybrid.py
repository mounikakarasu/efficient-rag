from app.hybrid import rrf_fuse


def test_rrf_fuse_prefers_items_ranked_high_in_both():
    dense = [10, 20, 30]
    sparse = [10, 40, 50]
    fused = rrf_fuse([dense, sparse], k=60)
    assert fused[0] == 10


def test_rrf_fuse_includes_all_ids():
    fused = rrf_fuse([[1, 2], [2, 3]], k=60)
    assert set(fused) == {1, 2, 3}


def test_rrf_fuse_rewards_higher_rank():
    fused = rrf_fuse([[1, 2, 3]], k=60)
    assert fused == [1, 2, 3]
