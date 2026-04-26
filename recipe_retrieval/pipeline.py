"""End-to-end: normalize -> candidates -> rank."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from recipe_retrieval.corpus import iter_recipe_files
from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.normalize import IdentityNormalizer, IngredientNormalizer
from recipe_retrieval.rankers import PenaltyConfig, RankerName, rank
from recipe_retrieval.schema import DetectedIngredient, NormalizedIngredient, RetrievalResult
from recipe_retrieval.text import ingredient_key

DetectionInput = Union[list[DetectedIngredient], list[dict]]


def build_index_from_paths(paths: list[str | Path]) -> RecipeIndex:
    if not paths:
        raise ValueError("At least one recipe corpus path is required")
    records = list(iter_recipe_files(list(paths)))
    if not records:
        raise ValueError(
            "No recipe records were loaded from the provided path(s). "
            "Ensure file format is JSON array or JSONL with recipe objects."
        )
    return RecipeIndex.build(records)


def _candidate_ids(index: RecipeIndex, normalized: list[NormalizedIngredient]) -> set[str]:
    qkeys = {ingredient_key(n.canonical) for n in normalized if ingredient_key(n.canonical)}
    if not qkeys:
        return set()
    c = index.candidate_recipe_ids(qkeys)
    if not c and len(index) > 0:
        return index.all_recipe_ids()
    return c


def retrieve(
    detected: DetectionInput,
    *,
    index: RecipeIndex,
    ranker: RankerName = "penalty_aware",
    k: int = 5,
    normalizer: IngredientNormalizer | None = None,
    penalty_config: PenaltyConfig | None = None,
) -> RetrievalResult:
    """
    Main integration entry: list of detections (or dicts with ingredient/confidence) -> top-k.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if detected and isinstance(detected[0], dict):
        norm_list: list[DetectedIngredient] = [
            DetectedIngredient(ingredient=str(d["ingredient"]), confidence=float(d["confidence"]))
            for d in detected  # type: ignore[misc]
        ]
    else:
        norm_list = list(detected)  # type: ignore[arg-type]
    nrm = (normalizer or IdentityNormalizer()).normalize(norm_list)
    cands = _candidate_ids(index, nrm)
    top = rank(ranker, index, nrm, cands if cands else None, k, penalty_config)
    return RetrievalResult(
        query=norm_list,
        normalized=nrm,
        top_k=top,
        ranker_name=ranker,
        k=k,
    )
