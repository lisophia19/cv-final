"""Baseline rankers: overlap, confidence-weighted, penalty-aware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.schema import NormalizedIngredient, RankedRecipe, ScoreBreakdown
from recipe_retrieval.text import ingredient_key

RankerName = Literal["overlap", "confidence_weighted", "penalty_aware"]


@dataclass
class PenaltyConfig:
    missing_penalty: float = 0.12
    """Subtracted per recipe ingredient (normalized key) not covered by query (capped)."""

    missing_cap: float = 1.5
    """Max total missing penalty to avoid one huge recipe being crushed."""

    use_query_weight_sum_norm: bool = True
    """Normalize confidence-weighted sum by sum of query weights."""


def _query_map(ingredients: list[NormalizedIngredient]) -> dict[str, float]:
    m: dict[str, float] = {}
    for n in ingredients:
        k = ingredient_key(n.canonical)
        if not k:
            continue
        w = float(n.weight)
        if k in m:
            m[k] = max(m[k], w)
        else:
            m[k] = w
    return m


def rank_overlap(
    index: RecipeIndex,
    query: list[NormalizedIngredient],
    candidate_ids: set[str] | None,
    k: int,
) -> list[RankedRecipe]:
    qmap = _query_map(query)
    qkeys = set(qmap.keys())
    cands = candidate_ids if candidate_ids is not None else index.all_recipe_ids()
    if not cands and len(index) > 0:
        cands = index.all_recipe_ids()
    scored: list[RankedRecipe] = []
    for rid in cands:
        r_ings = index.recipe_ingredient_keys(rid)
        inter = qkeys & r_ings
        if not qkeys and not r_ings:
            score = 0.0
        elif not qkeys:
            score = 0.0
        else:
            score = len(inter) / max(len(qkeys), 1)
        rec = index.recipes[rid]
        br = ScoreBreakdown(
            total=score,
            terms={"intersection": float(len(inter)), "|query|": float(len(qkeys))},
        )
        scored.append(
            RankedRecipe(recipe_id=rid, title=rec.title, score=score, breakdown=br)
        )
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]


def rank_confidence_weighted(
    index: RecipeIndex,
    query: list[NormalizedIngredient],
    candidate_ids: set[str] | None,
    k: int,
    cfg: PenaltyConfig | None = None,
) -> list[RankedRecipe]:
    cfg = cfg or PenaltyConfig()
    qmap = _query_map(query)
    qkeys = set(qmap.keys())
    sum_w = sum(qmap.values()) or 1.0
    cands = candidate_ids if candidate_ids is not None else index.all_recipe_ids()
    if not cands and len(index) > 0:
        cands = index.all_recipe_ids()
    out: list[RankedRecipe] = []
    for rid in cands:
        r_ings = index.recipe_ingredient_keys(rid)
        matched = qkeys & r_ings
        raw = sum(qmap.get(x, 0.0) for x in matched)
        score = raw / sum_w if cfg.use_query_weight_sum_norm else raw
        rec = index.recipes[rid]
        br = ScoreBreakdown(
            total=score,
            terms={"raw_weighted_match": raw, "sum_query_weights": sum_w, "|matched|": float(len(matched))},
        )
        out.append(RankedRecipe(recipe_id=rid, title=rec.title, score=score, breakdown=br))
    out.sort(key=lambda x: (x.score, x.title), reverse=True)
    return out[:k]


def rank_penalty_aware(
    index: RecipeIndex,
    query: list[NormalizedIngredient],
    candidate_ids: set[str] | None,
    k: int,
    cfg: PenaltyConfig | None = None,
) -> list[RankedRecipe]:
    cfg = cfg or PenaltyConfig()
    qmap = _query_map(query)
    qkeys = set(qmap.keys())
    sum_w = sum(qmap.values()) or 1.0
    cands = candidate_ids if candidate_ids is not None else index.all_recipe_ids()
    if not cands and len(index) > 0:
        cands = index.all_recipe_ids()
    out: list[RankedRecipe] = []
    for rid in cands:
        r_ings = index.recipe_ingredient_keys(rid)
        matched = qkeys & r_ings
        raw = sum(qmap.get(x, 0.0) for x in matched)
        base = raw / sum_w
        # Missing recipe ingredients not in query
        missing = r_ings - qkeys
        miss_pen = min(len(missing) * cfg.missing_penalty, cfg.missing_cap)
        score = max(base - miss_pen, 0.0)
        rec = index.recipes[rid]
        br = ScoreBreakdown(
            total=score,
            terms={
                "base_overlap_weighted": base,
                "missing_count": float(len(missing)),
                "missing_penalty_applied": miss_pen,
            },
            details={"missing_keys_sample": list(sorted(missing))[:20]},
        )
        out.append(RankedRecipe(recipe_id=rid, title=rec.title, score=score, breakdown=br))
    out.sort(key=lambda x: (x.score, -len(index.recipe_ingredient_keys(x.recipe_id))), reverse=True)
    return out[:k]


def rank(
    name: RankerName,
    index: RecipeIndex,
    query: list[NormalizedIngredient],
    candidate_ids: set[str] | None,
    k: int,
    penalty_config: PenaltyConfig | None = None,
) -> list[RankedRecipe]:
    if name == "overlap":
        return rank_overlap(index, query, candidate_ids, k)
    if name == "confidence_weighted":
        return rank_confidence_weighted(index, query, candidate_ids, k, penalty_config)
    if name == "penalty_aware":
        return rank_penalty_aware(index, query, candidate_ids, k, penalty_config)
    raise ValueError(f"Unknown ranker: {name}")
