"""Inverted index: ingredient key -> set of recipe ids, plus recipe store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from recipe_retrieval.corpus import RecipeRecord
from recipe_retrieval.text import ingredient_key

if TYPE_CHECKING:
    pass


@dataclass
class RecipeIndex:
    """In-memory index for candidate generation."""

    recipes: dict[str, RecipeRecord] = field(default_factory=dict)
    inverted: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, records: list[RecipeRecord]) -> "RecipeIndex":
        idx = cls()
        for r in records:
            if r.recipe_id in idx.recipes:
                raise ValueError(f"Duplicate recipe_id: {r.recipe_id}")
            idx.recipes[r.recipe_id] = r
            seen_keys: set[str] = set()
            for line in r.ingredients:
                k = ingredient_key(line)
                if not k:
                    continue
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                idx.inverted.setdefault(k, set()).add(r.recipe_id)
        return idx

    def recipe_ingredient_keys(self, recipe_id: str) -> set[str]:
        r = self.recipes[recipe_id]
        return {ingredient_key(x) for x in r.ingredients if ingredient_key(x)}

    def candidate_recipe_ids(self, query_keys: set[str]) -> set[str]:
        out: set[str] = set()
        for k in query_keys:
            if k in self.inverted:
                out |= self.inverted[k]
        if not out and query_keys:
            # Fallback: union of all recipes that share any token overlap is expensive;
            # for empty OR intersection mode we still want some candidates: use union of all buckets
            for k in query_keys:
                out |= self.inverted.get(k, set())
        return out

    def all_recipe_ids(self) -> set[str]:
        return set(self.recipes.keys())

    def __len__(self) -> int:
        return len(self.recipes)

    def __iter__(self) -> Iterator[RecipeRecord]:
        return iter(self.recipes.values())
