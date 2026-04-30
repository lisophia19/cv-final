"""
Recipe retrieval and ranking (Martin). Stable exports for integration.
"""

from recipe_retrieval.corpus import RecipeRecord, load_recipes_auto
from recipe_retrieval.eval import (
    AblationRun,
    evaluate_cases,
    load_eval_cases,
    run_ablation,
    write_artifact,
)
from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.normalize import AliasFileNormalizer, IdentityNormalizer
from recipe_retrieval.pipeline import build_index_from_paths, retrieve
from recipe_retrieval.rankers import PenaltyConfig, RankerName, rank
from recipe_retrieval.schema import (
    DetectedIngredient,
    NormalizedIngredient,
    RankedRecipe,
    RetrievalResult,
    ScoreBreakdown,
)
from recipe_retrieval.integrate import get_normalizer_for_project, retrieve_with_reconciled_vocab

__all__ = [
    "AliasFileNormalizer",
    "IdentityNormalizer",
    "DetectedIngredient",
    "NormalizedIngredient",
    "RankedRecipe",
    "RetrievalResult",
    "ScoreBreakdown",
    "RecipeRecord",
    "load_recipes_auto",
    "RecipeIndex",
    "build_index_from_paths",
    "retrieve",
    "RankerName",
    "PenaltyConfig",
    "rank",
    "evaluate_cases",
    "run_ablation",
    "load_eval_cases",
    "write_artifact",
    "AblationRun",
    "get_normalizer_for_project",
    "retrieve_with_reconciled_vocab",
]
