"""CLI: retrieval demo and ablation eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recipe_retrieval.eval import load_eval_cases, run_ablation, write_artifact
from recipe_retrieval.normalize import AliasFileNormalizer, IdentityNormalizer
from recipe_retrieval.pipeline import build_index_from_paths, retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Recipe retrieval: demo or eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run retrieval on a JSON list of detections")
    d.add_argument("--recipes", required=True, nargs="+", help="JSON/JSONL recipe file(s)")
    d.add_argument("--query", required=True, help="JSON file: list of {ingredient, confidence}")
    d.add_argument("--ranker", default="penalty_aware")
    d.add_argument("-k", type=int, default=5)
    d.add_argument("--alias", help="optional JSON map raw_lower -> canonical")

    e = sub.add_parser("eval", help="Run ablation and write results under --out")
    e.add_argument("--recipes", required=True, nargs="+", help="Recipe corpus file(s)")
    e.add_argument("--cases", required=True, help="JSONL eval cases (see fridge_data/eval_cases.jsonl)")
    e.add_argument("--out", default="runs/retrieval_eval", help="Output directory for artifacts")
    e.add_argument("-k", type=int, default=5, dest="k_max")
    e.add_argument("--alias", help="optional JSON map for AliasFileNormalizer")

    args = p.parse_args()
    if args.cmd == "demo":
        index = build_index_from_paths(args.recipes)
        q = json.loads(Path(args.query).read_text(encoding="utf-8"))
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
        normalizer = AliasFileNormalizer(args.alias) if args.alias else IdentityNormalizer()
        ablations = run_ablation(
            index,
            cases,
            k_max=args.k_max,
            normalizer=normalizer,
        )
        f = write_artifact(
            args.out,
            ablations=ablations,
            extra_meta={"cases_path": str(args.cases), "recipes": list(args.recipes)},
        )
        for a in ablations:
            print(a.ranker, a.metrics)
        print("Wrote", f)


if __name__ == "__main__":
    main()
