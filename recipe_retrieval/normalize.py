"""
Ingredient normalization hook.

Replace or extend with teammate vocabulary/alias table when frozen.
`IdentityNormalizer` is the default; `AliasFileNormalizer` loads JSON:
  { "raw_lower": "canonical", ... }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from recipe_retrieval.schema import DetectedIngredient, NormalizedIngredient


class IngredientNormalizer(Protocol):
    def normalize(self, detected: list[DetectedIngredient]) -> list[NormalizedIngredient]: ...


def _default_weight(d: DetectedIngredient) -> float:
    return float(d.confidence)


class IdentityNormalizer:
    """No alias map: canonical == stripped raw string, weight = confidence."""

    def normalize(self, detected: list[DetectedIngredient]) -> list[NormalizedIngredient]:
        out: list[NormalizedIngredient] = []
        seen: set[str] = set()
        for d in detected:
            raw = d.ingredient.strip()
            if not raw:
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                NormalizedIngredient(
                    canonical=raw,
                    source_raw=raw,
                    weight=_default_weight(d),
                )
            )
        return out


class AliasFileNormalizer:
    """Map raw (case-insensitive) to canonical; unknown tokens pass through as canonical=raw."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Alias file must be a JSON object mapping raw -> canonical")
        self._alias: dict[str, str] = {str(k).lower().strip(): str(v).strip() for k, v in data.items()}

    def normalize(self, detected: list[DetectedIngredient]) -> list[NormalizedIngredient]:
        out: list[NormalizedIngredient] = []
        by_canon: dict[str, float] = {}
        for d in detected:
            raw = d.ingredient.strip()
            if not raw:
                continue
            key = raw.lower()
            canonical = self._alias.get(key, raw)
            w = _default_weight(d)
            if canonical in by_canon:
                by_canon[canonical] = max(by_canon[canonical], w)
            else:
                by_canon[canonical] = w
        for canonical, w in by_canon.items():
            out.append(
                NormalizedIngredient(
                    canonical=canonical,
                    source_raw=canonical,
                    weight=w,
                )
            )
        return out
