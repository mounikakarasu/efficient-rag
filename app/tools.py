from __future__ import annotations

import ast
import operator
import re

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


_WORD_OPS = (
    ("divided by", "/"),
    ("times", "*"),
    ("multiplied by", "*"),
    ("plus", "+"),
    ("minus", "-"),
)
_EXPR_RE = re.compile(r"[0-9.][0-9.\s+\-*/%()]*")


def extract_expression(text: str) -> str | None:
    t = text.lower().replace("^", "**")
    for word, sym in _WORD_OPS:
        t = t.replace(word, sym)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)

    candidates = [
        m.group().strip()
        for m in _EXPR_RE.finditer(t)
        if re.search(r"[+\-*/%]", m.group()) and re.search(r"\d", m.group())
    ]
    if not candidates:
        return None
    expr = max(candidates, key=len).strip()
    if len(re.findall(r"\d+(?:\.\d+)?", expr)) < 2:
        return None
    try:
        calculate(expr)
    except (ValueError, SyntaxError, ZeroDivisionError, KeyError):
        return None
    return expr
