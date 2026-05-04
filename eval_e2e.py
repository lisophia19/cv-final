"""
End-to-end evaluation script for the recipe recommender pipeline.

Runs detection + retrieval over labeled fridge/kitchen photos and reports:
  - Detection: precision / recall / F1 vs. true_ingredients
  - Retrieval: top-1 / top-3 / top-5 match rate vs. reasonable_recipes

Usage:
    python eval_e2e.py
    python eval_e2e.py --labels eval_data/labels.jsonl --out runs/e2e_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from PIL import Image
from ultralytics import YOLO

from recipe_retrieval.integrate import retrieve_with_reconciled_vocab
from recipe_retrieval.pipeline import build_index_from_paths
from recipe_retrieval.rankers import PenaltyConfig

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = BASE_DIR / "eval_data" / "labels.jsonl"
DEFAULT_OUT_DIR = BASE_DIR / "runs" / "e2e_eval"


def _resolve_yolo_weights(base: Path) -> Path:
    for p in (
        base / "yolo" / "ingredients_yolo11n" / "weights" / "best.pt",
        base / "runs" / "ingredients_yolo11n" / "weights" / "best.pt",
    ):
        if p.exists():
            return p
    return base / "yolo" / "ingredients_yolo11n" / "weights" / "best.pt"


MODEL_PATH = _resolve_yolo_weights(BASE_DIR)
RECIPE_PATH = BASE_DIR / "fridge_data" / "sample_recipes.jsonl"
ALIAS_PATH = BASE_DIR / "fridge_data" / "team_ingredient_alias.json"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_TOP_K = 5


def normalize(s: str) -> str:
    return s.lower().strip()


def ingredient_match(detected: list[str], true: list[str]) -> tuple[int, int, int]:
    """Greedy 1-to-1 substring match. Returns (true_positives, false_positives, false_negatives)."""
    det = [normalize(d) for d in detected]
    tru = [normalize(t) for t in true]

    matched_true: set[int] = set()
    matched_det: set[int] = set()

    for i, d in enumerate(det):
        for j, t in enumerate(tru):
            if j in matched_true or i in matched_det:
                continue
            if t in d or d in t:
                matched_true.add(j)
                matched_det.add(i)
                break

    tp = len(matched_det)
    fp = len(det) - tp
    fn = len(tru) - len(matched_true)
    return tp, fp, fn


def recipe_match(retrieved_titles: list[str], reasonable: list[str], k: int) -> bool:
    """Does any of the top-k retrieved recipes match a reasonable recipe (substring, either direction)?"""
    reasonable_lower = [normalize(r) for r in reasonable]
    for title in retrieved_titles[:k]:
        title_lower = normalize(title)
        for r in reasonable_lower:
            if r in title_lower or title_lower in r:
                return True
    return False


def evaluate_one(
    model: YOLO,
    index,
    image_path: Path,
    true_ingredients: list[str],
    reasonable_recipes: list[str],
    *,
    conf_threshold: float,
    top_k: int,
    ranker: str,
    penalty_config: PenaltyConfig,
) -> dict | None:
    if not image_path.exists():
        return None

    img = Image.open(image_path).convert("RGB")
    results = model.predict(source=img, conf=conf_threshold, save=False, verbose=False)
    result = results[0]

    best: dict[str, float] = {}
    for box in result.boxes:
        name = model.names[int(box.cls.item())]
        conf = float(box.conf.item())
        if name not in best or conf > best[name]:
            best[name] = conf

    detected = list(best.keys())
    detected_with_conf = [
        {"ingredient": name, "confidence": round(conf, 3)}
        for name, conf in best.items()
    ]

    tp, fp, fn = ingredient_match(detected, true_ingredients)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    retrieval = retrieve_with_reconciled_vocab(
        detected_with_conf,
        index=index,
        ranker=ranker,
        k=top_k,
        alias_path=ALIAS_PATH if ALIAS_PATH.exists() else None,
        penalty_config=penalty_config,
    )

    retrieved_titles: list[str] = []
    for ranked in retrieval.top_k:
        rec = index.recipes.get(ranked.recipe_id)
        if rec is not None:
            retrieved_titles.append(rec.title)

    return {
        "image": str(image_path.relative_to(BASE_DIR)),
        "n_true": len(true_ingredients),
        "n_detected": len(detected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "top1_match": recipe_match(retrieved_titles, reasonable_recipes, 1),
        "top3_match": recipe_match(retrieved_titles, reasonable_recipes, 3),
        "top5_match": recipe_match(retrieved_titles, reasonable_recipes, 5),
        "detected_ingredients": detected,
        "retrieved_recipes": retrieved_titles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to labels.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for results")
    parser.add_argument("--ranker", default="penalty_aware", help="ranker to use for retrieval")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="retrieval cutoff for suggestions")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--missing-penalty", type=float, default=0.12, help="penalty per missing ingredient")
    parser.add_argument("--missing-cap", type=float, default=1.5, help="maximum total missing penalty")
    parser.add_argument(
        "--no-query-weight-norm",
        action="store_true",
        help="disable confidence-weight normalization in weighted rankers",
    )
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"Labels file not found: {args.labels}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"YOLO weights not found: {MODEL_PATH}")
    if not RECIPE_PATH.exists():
        raise SystemExit(f"Recipe corpus not found: {RECIPE_PATH}")

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Building recipe index from {RECIPE_PATH}")
    index = build_index_from_paths([RECIPE_PATH])

    cases: list[dict] = []
    with open(args.labels) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))

    print(f"Evaluating {len(cases)} cases\n")
    penalty_cfg = PenaltyConfig(
        missing_penalty=args.missing_penalty,
        missing_cap=args.missing_cap,
        use_query_weight_sum_norm=(not args.no_query_weight_norm),
    )

    per_image: list[dict] = []
    for case in cases:
        image_path = (BASE_DIR / case["image"]).resolve()
        result = evaluate_one(
            model,
            index,
            image_path,
            case["true_ingredients"],
            case["reasonable_recipes"],
            conf_threshold=args.conf_threshold,
            top_k=args.top_k,
            ranker=args.ranker,
            penalty_config=penalty_cfg,
        )
        if result is None:
            print(f"  SKIP (image not found): {case['image']}")
            continue
        per_image.append(result)
        print(
            f"  {result['image']}: "
            f"P={result['precision']:.2f} R={result['recall']:.2f} F1={result['f1']:.2f} "
            f"top1={int(result['top1_match'])} top3={int(result['top3_match'])} top5={int(result['top5_match'])}"
        )

    if not per_image:
        print("\nNo valid evaluation cases.")
        return

    aggregate = {
        "n_images": len(per_image),
        "mean_precision": round(mean(r["precision"] for r in per_image), 3),
        "mean_recall": round(mean(r["recall"] for r in per_image), 3),
        "mean_f1": round(mean(r["f1"] for r in per_image), 3),
        "top1_rate": round(mean(r["top1_match"] for r in per_image), 3),
        "top3_rate": round(mean(r["top3_match"] for r in per_image), 3),
        "top5_rate": round(mean(r["top5_match"] for r in per_image), 3),
    }

    print("\n=== AGGREGATE ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    args.out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = args.out / f"e2e_eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "model": str(MODEL_PATH.relative_to(BASE_DIR)),
                    "recipe_corpus": str(RECIPE_PATH.relative_to(BASE_DIR)),
                    "conf_threshold": args.conf_threshold,
                    "top_k": args.top_k,
                    "ranker": args.ranker,
                    "missing_penalty": args.missing_penalty,
                    "missing_cap": args.missing_cap,
                    "use_query_weight_sum_norm": not args.no_query_weight_norm,
                },
                "aggregate": aggregate,
                "per_image": per_image,
            },
            f,
            indent=2,
        )

    csv_path = args.out / f"e2e_eval_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "image", "n_true", "n_detected", "tp", "fp", "fn",
            "precision", "recall", "f1",
            "top1_match", "top3_match", "top5_match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_image:
            writer.writerow(row)

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
