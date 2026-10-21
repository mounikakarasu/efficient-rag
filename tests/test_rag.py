from pathlib import Path

import pytest

from app.config import config
from app.rag import answer_question, build_messages
from app.ingest import build_index
from app.retriever import RetrievedChunk, Retriever

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_AVAILABLE = config.llm_model_path.exists()


def test_build_messages_includes_context_and_question():
    chunks = [
        RetrievedChunk(id=0, text="Venus is the hottest planet.",
                       source="solar_system.txt", chunk_index=0, score=0.9),
    ]
    messages = build_messages("Which planet is hottest?", chunks)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "Venus is the hottest planet." in user
    assert "solar_system.txt" in user
    assert "Which planet is hottest?" in user


def test_build_messages_handles_no_context():
    messages = build_messages("anything", [])
    assert "no relevant context" in messages[1]["content"].lower()


@pytest.mark.slow
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="GGUF model not downloaded")
def test_generation_is_grounded(tmp_path):
    index_dir = tmp_path / "index"
    build_index(FIXTURES, index_dir, config)
    retriever = Retriever.load(index_dir)

    result = answer_question(
        "How many vacation days do full-time employees get per year?",
        retriever=retriever,
    )
    assert "20" in result.answer
    assert result.sources
    assert result.latency_ms > 0
    assert result.completion_tokens > 0
    assert any(s["source"] == "company_handbook.md" for s in result.sources)
