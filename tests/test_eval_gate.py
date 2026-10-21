import json

import numpy as np

from app.build_index import build_serving_index
from app.config import config
from app.eval_metrics import recall_at_k
from app.hybrid import HybridRetriever
from app.embed_corpus import embed_corpus
from app.sparse import build_bm25
from app.vector_index import load_chunk_meta

CORPUS = [
    ("py-file", "How to read a file line by line in Python using readline and a for loop over the file object."),
    ("py-sort", "Sorting a list of dictionaries by a key in Python using sorted with a lambda key function."),
    ("java-str", "Reversing a string in Java using the StringBuilder reverse method efficiently."),
    ("sql-join", "Writing an inner join in SQL to combine rows from two tables on a matching key column."),
    ("css-center", "Centering a div horizontally and vertically with CSS flexbox using justify-content and align-items."),
]

QUERIES = [
    ("read a text file line by line in python", "py-file"),
    ("sort list of dicts by key python", "py-sort"),
    ("reverse a string in java", "java-str"),
    ("sql inner join two tables", "sql-join"),
    ("center a div with flexbox", "css-center"),
]

RECALL_GATE = 0.8


def test_retrieval_quality_gate(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as f:
        for doc_id, text in CORPUS:
            f.write(json.dumps({"id": doc_id, "source": "so", "url": f"http://x/{doc_id}", "text": text * 6}) + "\n")

    emb = tmp_path / "emb"
    embed_corpus(corpus, emb, config)
    build_serving_index(emb, "hnsw")
    build_bm25(emb)

    retriever = HybridRetriever.load(emb, rerank_enabled=False)
    meta = load_chunk_meta(emb)

    recalls = []
    for query, relevant in QUERIES:
        hits = retriever.search(query, top_k=3)
        doc_ids = [meta[h.id]["doc_id"] for h in hits]
        recalls.append(recall_at_k(doc_ids, relevant, 3))

    assert float(np.mean(recalls)) >= RECALL_GATE
