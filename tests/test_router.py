import pytest

from app.config import config
from app.router import (
    CALCULATE,
    DIRECT_ANSWER,
    RETRIEVE_AND_ANSWER,
    Router,
    classify_intent,
)
from app.router_langgraph import LangGraphRouter

ROUTING_CASES = [
    ("What is 12 * 7?", CALCULATE),
    ("calculate 100 / 4", CALCULATE),
    ("How many moons does Mars have?", RETRIEVE_AND_ANSWER),
    ("Tell me about the equipment budget", RETRIEVE_AND_ANSWER),
    ("hello", DIRECT_ANSWER),
    ("hi there", DIRECT_ANSWER),
    ("thanks", DIRECT_ANSWER),
    ("who are you", DIRECT_ANSWER),
]


@pytest.mark.parametrize("query,expected", ROUTING_CASES)
def test_classify_intent(query, expected):
    assert classify_intent(query) == expected


def test_plain_router_calculate_dispatch():
    result = Router().route("What is 12 * 7?")
    assert result.intent == CALCULATE
    assert result.tool == "calculator"
    assert result.detail["result"] == 84
    assert "84" in result.answer
    assert result.latency_ms >= 0


def test_langgraph_router_calculate_dispatch():
    result = LangGraphRouter().route("What is 12 * 7?")
    assert result.intent == CALCULATE
    assert result.detail["result"] == 84
    assert "84" in result.answer


@pytest.mark.parametrize("query,expected", ROUTING_CASES)
def test_backends_agree_on_intent(query, expected):
    plain_intent = classify_intent(query)
    assert plain_intent == expected
    if expected == CALCULATE:
        plain = Router().route(query)
        graph = LangGraphRouter().route(query)
        assert plain.intent == graph.intent
        assert plain.answer == graph.answer
