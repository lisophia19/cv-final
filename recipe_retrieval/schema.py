"""I/O types for recipe retrieval and ranking (stable integration contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectedIngredient:
    """Output shape aligned with `test.py` post-processing: ingredient name + model confidence."""

    ingredient: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.ingredient, str):
            raise TypeError("ingredient must be str")
        c = float(self.confidence)
        if c < 0.0:
            raise ValueError("confidence must be non-negative")


@dataclass(frozen=True)
class NormalizedIngredient:
    """After vocabulary / alias map (teammate-owned normalization layer)."""

    canonical: str
    source_raw: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-recipe score decomposition for rubrics and debugging."""

    total: float
    terms: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedRecipe:
    """One ranked candidate."""

    recipe_id: str
    title: str
    score: float
    breakdown: ScoreBreakdown


@dataclass(frozen=True)
class RetrievalResult:
    """Stable output for integrators: ranked list + echo of query for logging."""

    query: list[DetectedIngredient]
    normalized: list[NormalizedIngredient]
    top_k: list[RankedRecipe]
    ranker_name: str
    k: int
