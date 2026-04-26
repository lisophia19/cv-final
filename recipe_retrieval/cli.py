"""CLI: retrieval demo and ablation eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recipe_retrieval.eval import load_eval_cases, run_ablation, write_artifact
from recipe_retrieval.normalize import AliasFileNormalizer, IdentityNormalizer
from recipe_retrieval.pipeline import build_index_from_paths, retrieve
from recipe_retrieval.rankers import PenaltyConfig


def _build_penalty_config(args: argparse.Namespace) -> PenaltyConfig:
    cfg = PenaltyConfig()
    if args.missing_penalty is not None:
        cfg.missing_penalty = float(args.missing_penalty)
    if args.missing_cap is not None:
        cfg.missing_cap = float(args.missing_cap)
    if args.no_query_weight_norm:
        cfg.use_query_weight_sum_norm = False
    return cfg


def _load_tuning_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Tuning config must be a JSON object")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Recipe retrieval: demo or eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run retrieval on a JSON list of detections")
    d.add_argument("--recipes", required=True, nargs="+", help="JSON/JSONL recipe file(s)")
    d.add_argument("--query", required=True, help="JSON file: list of {ingredient, confidence}")
    d.add_argument("--ranker", default="penalty_aware")
    d.add_argument("-k", type=int, default=5)
    d.add_argument("--alias", help="optional JSON map raw_lower -> canonical")
    d.add_argument("--missing-penalty", type=float, help="penalty per missing recipe ingredient")
    d.add_argument("--missing-cap", type=float, help="maximum total missing penalty")
    d.add_argument("--no-query-weight-norm", action="store_true", help="disable query-weight normalization")
    d.add_argument("--tuning-config", help="optional JSON config for ranker/tuning parameters")

    e = sub.add_parser("eval", help="Run ablation and write results under --out")
    e.add_argument("--recipes", required=True, nargs="+", help="Recipe corpus file(s)")
    e.add_argument("--cases", required=True, help="JSONL eval cases (see fridge_data/eval_cases.jsonl)")
    e.add_argument("--out", default="runs/retrieval_eval", help="Output directory for artifacts")
    e.add_argument("-k", type=int, default=5, dest="k_max")
    e.add_argument("--alias", help="optional JSON map for AliasFileNormalizer")
    e.add_argument(
        "--rankers",
        nargs="+",
        default=["overlap", "confidence_weighted", "penalty_aware"],
        help="rankers to evaluate (space-separated)",
    )
    e.add_argument("--missing-penalty", type=float, help="penalty per missing recipe ingredient")
    e.add_argument("--missing-cap", type=float, help="maximum total missing penalty")
    e.add_argument("--no-query-weight-norm", action="store_true", help="disable query-weight normalization")
    e.add_argument("--tuning-config", help="optional JSON config for ranker/tuning parameters")

    args = p.parse_args()
    cfg_from_file = _load_tuning_config(args.tuning_config)
    if args.cmd == "demo":
        index = build_index_from_paths(args.recipes)
        q = json.loads(Path(args.query).read_text(encoding="utf-8"))
        # Allow config file defaults while preserving CLI explicit overrides.
        if "ranker" in cfg_from_file and args.ranker == "penalty_aware":
            args.ranker = cfg_from_file["ranker"]
        if "missing_penalty" in cfg_from_file and args.missing_penalty is None:
            args.missing_penalty = cfg_from_file["missing_penalty"]
        if "missing_cap" in cfg_from_file and args.missing_cap is None:
            args.missing_cap = cfg_from_file["missing_cap"]
        if "use_query_weight_sum_norm" in cfg_from_file and not args.no_query_weight_norm:
            args.no_query_weight_norm = not bool(cfg_from_file["use_query_weight_sum_norm"])
        penalty_cfg = _build_penalty_config(args)
        normalizer: IdentityNormalizer | AliasFileNormalizer
        if args.alias:
            normalizer = AliasFileNormalizer(args.alias)
        else:
            normalizer = IdentityNormalizer()
        res = retrieve(
            q,
            index=index,
            ranker=args.ranker,  # type: ignore[arg-type]
            k=args.k,
            normalizer=normalizer,
            penalty_config=penalty_cfg,
        )
        print(json.dumps({  # stable summary for integrators
            "ranker": res.ranker_name,
            "k": res.k,
            "query": [{"ingredient": d.ingredient, "confidence": d.confidence} for d in res.query],
            "normalized": [{"canonical": n.canonical, "weight": n.weight} for n in res.normalized],
            "top_k": [
                {
                    "recipe_id": r.recipe_id,
                    "title": r.title,
                    "score": r.score,
                    "breakdown": {
                        "total": r.breakdown.total,
                        "terms": r.breakdown.terms,
                    },
                }
                for r in res.top_k
            ],
        }, indent=2))
    else:
        index = build_index_from_paths(args.recipes)
        cases = load_eval_cases(args.cases)
        if "rankers" in cfg_from_file and args.rankers == ["overlap", "confidence_weighted", "penalty_aware"]:
            args.rankers = list(cfg_from_file["rankers"])
        if "missing_penalty" in cfg_from_file and args.missing_penalty is None:
            args.missing_penalty = cfg_from_file["missing_penalty"]
        if "missing_cap" in cfg_from_file and args.missing_cap is None:
            args.missing_cap = cfg_from_file["missing_cap"]
        if "use_query_weight_sum_norm" in cfg_from_file and not args.no_query_weight_norm:
            args.no_query_weight_norm = not bool(cfg_from_file["use_query_weight_sum_norm"])
        penalty_cfg = _build_penalty_config(args)
        normalizer = AliasFileNormalizer(args.alias) if args.alias else IdentityNormalizer()
        ablations = run_ablation(
            index,
            cases,
            rankers=args.rankers,
            k_max=args.k_max,
            normalizer=normalizer,
            penalty_config=penalty_cfg,
        )
        f = write_artifact(
            args.out,
            ablations=ablations,
            extra_meta={
                "cases_path": str(args.cases),
                "recipes": list(args.recipes),
                "rankers": list(args.rankers),
                "missing_penalty": penalty_cfg.missing_penalty,
                "missing_cap": penalty_cfg.missing_cap,
                "use_query_weight_sum_norm": penalty_cfg.use_query_weight_sum_norm,
                "tuning_config": args.tuning_config,
            },
        )
        for a in ablations:
            print(a.ranker, a.metrics)
        print("Wrote", f)


if __name__ == "__main__":
    main()
