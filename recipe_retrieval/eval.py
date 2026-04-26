"""Top-k evaluation and ablation logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.normalize import IdentityNormalizer, IngredientNormalizer
from recipe_retrieval.pipeline import retrieve
from recipe_retrieval.rankers import PenaltyConfig, RankerName


@dataclass
class CaseResult:
    case_id: str
    gold_recipe_ids: list[str]
    hit_at_k: dict[int, bool]
    best_rank: int | None
    top_ids: list[str]


@dataclass
class AblationRun:
    ranker: str
    k_max: int
    case_results: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for c in self.case_results:
            d = asdict(c)
            d["hit_at_k"] = {str(ik): v for ik, v in c.hit_at_k.items()}
            rows.append(d)
        return {
            "ranker": self.ranker,
            "k_max": self.k_max,
            "metrics": self.metrics,
            "case_results": rows,
        }


def _best_rank(gold: Sequence[str], ordered_ids: list[str]) -> int | None:
    gset = set(gold)
    for i, rid in enumerate(ordered_ids, start=1):
        if rid in gset:
            return i
    return None


def _hits_at_k(gold: Sequence[str], ordered_ids: list[str], ks: list[int]) -> dict[int, bool]:
    gset = set(gold)
    out: dict[int, bool] = {}
    for kk in ks:
        top = set(ordered_ids[:kk])
        out[kk] = bool(gset & top)
    return out


def evaluate_cases(
    index: RecipeIndex,
    cases: list[dict[str, Any]],
    *,
    ranker: RankerName,
    k_max: int = 5,
    normalizer: IngredientNormalizer | None = None,
    penalty_config: PenaltyConfig | None = None,
) -> AblationRun:
    ks = [1, 3, 5] if k_max >= 5 else [k for k in [1, 3, 5] if k <= k_max]
    if not ks:
        ks = [k_max]
    case_results: list[CaseResult] = []
    for c in cases:
        cid = str(c.get("case_id", "unknown"))
        gold = [str(x) for x in c.get("gold_recipe_ids", [])]
        det = c.get("detected", [])
        res = retrieve(
            det,  # type: ignore[arg-type]
            index=index,
            ranker=ranker,
            k=k_max,
            normalizer=normalizer,
            penalty_config=penalty_config,
        )
        top_ids = [r.recipe_id for r in res.top_k]
        hit = _hits_at_k(gold, top_ids, ks)
        br = _best_rank(gold, top_ids)
        case_results.append(
            CaseResult(
                case_id=cid,
                gold_recipe_ids=gold,
                hit_at_k=hit,
                best_rank=br,
                top_ids=top_ids,
            )
        )
    n = max(len(cases), 1)
    metrics: dict[str, float] = {}
    for kk in ks:
        key = f"recall_at_{kk}"
        metrics[key] = sum(1 for cr in case_results if cr.hit_at_k.get(kk, False)) / n
    mrrs = []
    for cr in case_results:
        if cr.best_rank is not None:
            mrrs.append(1.0 / cr.best_rank)
        else:
            mrrs.append(0.0)
    metrics["mrr"] = sum(mrrs) / n
    return AblationRun(ranker=ranker, k_max=k_max, case_results=case_results, metrics=metrics)


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def run_ablation(
    index: RecipeIndex,
    cases: list[dict[str, Any]],
    *,
    rankers: Sequence[RankerName] | None = None,
    k_max: int = 5,
    normalizer: IngredientNormalizer | None = None,
    penalty_config: PenaltyConfig | None = None,
) -> list[AblationRun]:
    rankers = list(rankers) if rankers is not None else ["overlap", "confidence_weighted", "penalty_aware"]
    return [
        evaluate_cases(
            index,
            cases,
            ranker=r,
            k_max=k_max,
            normalizer=normalizer,
            penalty_config=penalty_config,
        )
        for r in rankers
    ]


def write_artifact(
    out_dir: str | Path,
    *,
    ablations: list[AblationRun],
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "created_utc": ts,
        "ablations": [a.to_dict() for a in ablations],
    }
    if extra_meta:
        payload["meta"] = extra_meta
    f = out / f"retrieval_eval_{ts}.json"
    f.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "latest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return f
