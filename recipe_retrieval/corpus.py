"""Load recipe corpora from JSON/JSONL for indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class RecipeRecord:
    recipe_id: str
    title: str
    ingredients: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


def _as_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    return [str(x)]


def _record_from_dict(d: dict[str, Any]) -> RecipeRecord:
    rid = d.get("id") or d.get("recipe_id") or d.get("uid")
    if rid is None:
        raise ValueError("Recipe must have 'id' or 'recipe_id'")
    title = d.get("title") or d.get("name") or ""
    ings = _as_str_list(d.get("ingredients") or d.get("ingredient_lines"))
    extra = {k: v for k, v in d.items() if k not in ("id", "recipe_id", "uid", "title", "name", "ingredients", "ingredient_lines")}
    return RecipeRecord(recipe_id=str(rid), title=str(title), ingredients=ings, extra=extra)


def load_recipes_from_jsonl(path: str | Path) -> list[RecipeRecord]:
    p = Path(path)
    out: list[RecipeRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not isinstance(d, dict):
            continue
        out.append(_record_from_dict(d))
    return out


def load_recipes_from_json_array(path: str | Path) -> list[RecipeRecord]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON array of recipe objects expected")
    return [_record_from_dict(d) for d in data if isinstance(d, dict)]


def load_recipes_auto(path: str | Path) -> list[RecipeRecord]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return load_recipes_from_json_array(p)
    return load_recipes_from_jsonl(p)


def iter_recipe_files(paths: list[str | Path]) -> Iterator[RecipeRecord]:
    for path in paths:
        yield from load_recipes_auto(path)
