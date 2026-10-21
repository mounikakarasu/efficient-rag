from app.sparse import BM25Index, tokenize


def test_tokenize():
    assert tokenize("Hello, World! foo_bar 123") == ["hello", "world", "foo", "bar", "123"]


def test_bm25_ranks_relevant_first():
    texts = [
        "python read a file line by line with readline",
        "java reverse a string with stringbuilder",
        "sort a list of numbers in descending order",
    ]
    idx = BM25Index.build(texts)
    results = idx.search("python file readline", top_k=2)
    assert results[0][0] == 0
    assert results[0][1] >= results[1][1]


def test_bm25_empty_query_returns_nothing():
    idx = BM25Index.build(["alpha beta gamma"])
    assert idx.search("", top_k=5) == []


def test_bm25_save_load(tmp_path):
    idx = BM25Index.build([
        "python file reading with readline",
        "java string reverse stringbuilder",
        "sql database query select where",
        "css flexbox responsive layout",
    ])
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    reloaded = BM25Index.load(path)
    assert reloaded.size == 4
    assert reloaded.search("python readline", top_k=1)[0][0] == 0
