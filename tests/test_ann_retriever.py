import json

from app.build_index import build_serving_index
from app.config import config
from app.embed_corpus import embed_corpus
from app.retriever import AnnRetriever


def _write_corpus(path):
    docs = [
        ("1", "http://x/1", "How to read a file line by line in Python using readline and a for loop. "),
        ("2", "http://x/2", "Sorting a list of dictionaries by a key in Python with sorted and lambda. "),
        ("3", "http://x/3", "Reversing a string in Java using StringBuilder reverse method. "),
    ]
    with path.open("w", encoding="utf-8") as f:
        for id_, url, text in docs:
            f.write(json.dumps({"id": id_, "source": "stackoverflow", "url": url, "text": text * 12}) + "\n")


def test_ann_retriever_end_to_end(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    emb = tmp_path / "emb"

    embed_corpus(corpus, emb, config)
    stats = build_serving_index(emb, "hnsw")
    assert (emb / "ann.index").exists()
    assert stats["vectors"] >= 3

    retriever = AnnRetriever.load(emb)
    hits = retriever.search("how do I read a file line by line in python", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert all(h.source.startswith("http") for h in hits)
    assert hits[0].source == "http://x/1"


def test_ann_retriever_empty_query(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    emb = tmp_path / "emb"
    embed_corpus(corpus, emb, config)
    build_serving_index(emb, "hnsw")
    assert AnnRetriever.load(emb).search("   ", top_k=3) == []
