from __future__ import annotations

import re

_PATTERNS = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[JWT]", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("[AWS_KEY]", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("[TOKEN]", re.compile(r"\b(?:sk|ghp|xox[baprs])[-_][A-Za-z0-9]{16,}\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("[PHONE]", re.compile(r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact(text: str) -> str:
    if not text:
        return text
    for replacement, pattern in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
