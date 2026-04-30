"""Shared text normalization for matching query ingredients to recipe lines."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def ingredient_key(s: str) -> str:
    """Lowercased, collapsed whitespace; light punctuation strip for keys."""
    t = s.strip().lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t
