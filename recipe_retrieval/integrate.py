"""
Dependency reconciliation (post-freeze).
When the team freezes detector output and shared vocabulary, wire them here
without changing the public `retrieve` signature.
"""

from __future__ import annotations

from pathlib import Path

from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.normalize import AliasFileNormalizer, IdentityNormalizer, IngredientNormalizer
from recipe_retrieval.rankers import PenaltyConfig, RankerName
from recipe_retrieval.schema import DetectedIngredient, RetrievalResult
from recipe_retrieval.pipeline import retrieve

# Default asset paths; override in integration layer or env
DEFAULT_ALIAS_PATH = "data/team_ingredient_alias.json"


def get_normalizer_for_project(alias_path: str | Path | None = None) -> IngredientNormalizer:
    """
    After vocabulary freeze, commit `data/team_ingredient_alias.json` to map
    detector strings -> recipe canonical tokens. Falls back to identity.
    """
    p = Path(alias_path or DEFAULT_ALIAS_PATH)
    if p.is_file():
        return AliasFileNormalizer(p)
    return IdentityNormalizer()


def retrieve_with_reconciled_vocab(
    detected: list[DetectedIngredient] | list[dict],
    *,
    index: RecipeIndex,
    ranker: RankerName = "penalty_aware",
    k: int = 5,
    alias_path: str | Path | None = None,
    penalty_config: PenaltyConfig | None = None,
) -> RetrievalResult:
    """
    Use this in end-to-end integration once alias map is checked in
    (or set alias_path to `data/sample_alias_map.json` for local tests).
    """
    nrm = get_normalizer_for_project(alias_path)
    return retrieve(
        detected,
        index=index,
        ranker=ranker,
        k=k,
        normalizer=nrm,
        penalty_config=penalty_config,
    )
