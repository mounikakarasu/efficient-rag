import pytest

from app.tools import calculate, extract_expression


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 2", 4),
        ("12 * 7", 84),
        ("100 / 4", 25),
        ("2 ** 10", 1024),
        ("(3 + 4) * 2", 14),
        ("10 % 3", 1),
        ("-5 + 8", 3),
    ],
)
def test_calculate(expr, expected):
    assert calculate(expr) == expected


def test_calculate_rejects_code_injection():
    with pytest.raises((ValueError, SyntaxError)):
        calculate("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        calculate("a + b")


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is 12 * 7?", "12 * 7"),
        ("calculate 100 / 4", "100 / 4"),
        ("3 x 4", "3*4"),
        ("what is 5 plus 6", "5 + 6"),
    ],
)
def test_extract_expression(query, expected):
    assert extract_expression(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "Tell me about Venus",
        "hello",
        "I have 3 cats",
    ],
)
def test_extract_expression_none_for_non_math(query):
    assert extract_expression(query) is None
