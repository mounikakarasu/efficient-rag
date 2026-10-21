import json

import numpy as np

from app.config import config
from app.embed_corpus import batch_size_for, embed_corpus, iter_chunks


def test_batch_size_for_prefers_larger_on_gpu():
    assert batch_size_for("cuda", config) == config.embed_batch_size_cuda
    assert batch_size_for("cpu", config) == config.embed_batch_size
    assert config.embed_batch_size_cuda >= config.embed_batch_size


def _write_corpus(path, docs):
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


def test_iter_chunks(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "1", "source": "so", "url": "u", "text": "word " * 400}])
    chunks = list(iter_chunks(corpus, chunk_size=200, overlap=40))
    assert len(chunks) > 1
    assert all(c["doc_id"] == "1" for c in chunks)
    assert chunks[0]["chunk_index"] == 0


def test_embed_corpus_outputs_aligned_vectors(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {"id": "1", "source": "so", "url": "u1", "text": "How do I sort a list in Python? " * 30},
            {"id": "2", "source": "so", "url": "u2", "text": "What is a Python decorator and how is it used? " * 30},
        ],
    )
    out = tmp_path / "emb"
    stats = embed_corpus(corpus, out, config)

    assert stats["chunks"] > 0
    assert stats["embedding_dim"] == 384

    chunk_lines = (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunk_lines) == stats["chunks"]

    vectors = np.fromfile(out / "vectors.f32", dtype="float32")
    assert vectors.size == stats["chunks"] * stats["embedding_dim"]

    matrix = vectors.reshape(stats["chunks"], stats["embedding_dim"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
