from pathlib import Path

import pytest

from app.config import config
from app.ingest import build_index
from app.retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    build_index(FIXTURES, index_dir, config)
    return Retriever.load(index_dir)


def test_search_returns_scored_sources(retriever):
    hits = retriever.search("How many moons does Mars have?", top_k=3)
    assert len(hits) == 3
    for hit in hits:
        assert hit.source
        assert isinstance(hit.score, float)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_retrieves_topically_relevant_chunk(retriever):
    top = retriever.search("Which planet is the hottest in the solar system?", top_k=1)[0]
    assert top.source == "solar_system.txt"
    assert "venus" in top.text.lower()


def test_search_routes_to_correct_document(retriever):
    top = retriever.search("How many vacation days do employees get?", top_k=1)[0]
    assert top.source == "company_handbook.md"


def test_top_k_is_respected(retriever):
    assert len(retriever.search("planets", top_k=2)) == 2
    assert len(retriever.search("planets", top_k=5)) == 5


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ", top_k=3) == []
