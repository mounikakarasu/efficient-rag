import json
from pathlib import Path

import pytest

from app.config import config
from app.ingest import build_index, chunk_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_chunk_text_basic_coverage():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for piece, start, end in chunks:
        assert piece.strip()
        assert end - start <= 200
        assert start < end
    assert chunks[0][1] == 0
    assert chunks[-1][2] <= len(text.strip())


def test_chunk_text_overlap_creates_shared_context():
    text = "abcdefghij " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert chunks[1][1] < chunks[0][2]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0, overlap=0)


def test_chunk_text_empty():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_build_index_end_to_end(tmp_path):
    index_dir = tmp_path / "index"
    stats = build_index(FIXTURES, index_dir, config)

    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["embedding_dim"] == 384

    assert (index_dir / "faiss.index").exists()
    assert (index_dir / "index_meta.json").exists()
    meta_lines = (index_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(meta_lines) == stats["chunks"]

    first = json.loads(meta_lines[0])
    assert {"id", "text", "source", "chunk_index", "char_start", "char_end"} <= first.keys()
