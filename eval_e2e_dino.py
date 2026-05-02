"""
End-to-end evaluation script for the FULL DINO pipeline (TokenCut + LoRA classifier).

Sibling to eval_e2e.py. Same evaluation methodology (substring-match per-ingredient
precision/recall/F1, top-k recipe match), but uses dino_pipeline.detect_and_classify
in place of YOLO.

Usage:
    python eval_e2e_dino.py
    python eval_e2e_dino.py --labels eval_data/labels.jsonl --out runs/e2e_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from PIL import Image

from dino_detect import load_model as load_detector
from dino_pipeline import detect_and_classify, load_classifier
from recipe_retrieval.integrate import retrieve_with_reconciled_vocab
from recipe_retrieval.pipeline import build_index_from_paths

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = BASE_DIR / "eval_data" / "labels.jsonl"
DEFAULT_OUT_DIR = BASE_DIR / "runs" / "e2e_eval"
RECIPE_PATH = BASE_DIR / "fridge_data" / "sample_recipes.jsonl"
ALIAS_PATH = BASE_DIR / "fridge_data" / "team_ingredient_alias.json"
TOP_K = 5


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
    detector,
    backbone,
    classifier,
    processor,
    class_names,
    index,
    image_path: Path,
    true_ingredients: list[str],
    reasonable_recipes: list[str],
    device: str,
) -> dict | None:
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")
    detections, _ = detect_and_classify(
        image, detector, backbone, classifier, processor, class_names,
        method="tokencut", device=device,
    )

    # Dedup ingredients by max confidence
    best: dict[str, float] = {}
    for d in detections:
        if d.ingredient not in best or d.confidence > best[d.ingredient]:
            best[d.ingredient] = d.confidence

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
        ranker="penalty_aware",
        k=TOP_K,
        alias_path=ALIAS_PATH if ALIAS_PATH.exists() else None,
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
    parser = argparse.ArgumentParser(description="Full DINO pipeline end-to-end evaluation")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to labels.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for results")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"Labels file not found: {args.labels}")
    if not RECIPE_PATH.exists():
        raise SystemExit(f"Recipe corpus not found: {RECIPE_PATH}")

    print(f"Loading detector (DINOv2-base, output_attentions=True)...")
    detector = load_detector(args.device)

    print(f"Loading classifier (DINOv2-base + LoRA + linear)...")
    backbone, classifier, processor, class_names = load_classifier(args.device)

    print(f"Building recipe index from {RECIPE_PATH}")
    index = build_index_from_paths([RECIPE_PATH])

    cases: list[dict] = []
    with open(args.labels) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))

    print(f"Evaluating {len(cases)} cases (DINO+DINO pipeline)\n")

    per_image: list[dict] = []
    for idx, case in enumerate(cases, 1):
        image_path = (BASE_DIR / case["image"]).resolve()
        result = evaluate_one(
            detector, backbone, classifier, processor, class_names,
            index, image_path, case["true_ingredients"], case["reasonable_recipes"],
            device=args.device,
        )
        if result is None:
            print(f"  [{idx}/{len(cases)}] SKIP (image not found): {case['image']}")
            continue
        per_image.append(result)
        print(
            f"  [{idx}/{len(cases)}] {result['image']}: "
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

    print("\n=== AGGREGATE (DINO+DINO pipeline) ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    args.out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = args.out / f"dino_e2e_eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "pipeline": "dino+dino",
                "config": {
                    "recipe_corpus": str(RECIPE_PATH.relative_to(BASE_DIR)),
                    "top_k": TOP_K,
                    "detection_method": "tokencut",
                },
                "aggregate": aggregate,
                "per_image": per_image,
            },
            f,
            indent=2,
        )

    csv_path = args.out / f"dino_e2e_eval_{timestamp}.csv"
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
