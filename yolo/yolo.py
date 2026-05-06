import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

# cv-final project root (parent of this yolo/ package)
REPO_ROOT = Path(__file__).resolve().parent.parent

def tune_model():
    # Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
    model = YOLO("yolo11n.pt")

    # Start training on your custom dataset
    data_yaml = REPO_ROOT / "ingredients_data" / "data.yaml"

    model.tune(
        data=str(data_yaml),
        epochs=50,  # epochs per trial
        iterations=10,  # number of tuning trials
        imgsz=800,
        batch=8,
        optimizer="AdamW",
        project=str(REPO_ROOT / "yolo" / "runs"),
        name="ingredients_tune11n",
    )

def train_model(
    epochs, imgsz, batch, patience, lr0, lrf, box, cls, weight_decay, workers=2, multi_scale=False
):
    """
    More complex training function with finetuning and specific hyperparam definition
    """
    data_yaml = REPO_ROOT / "ingredients_data" / "data.yaml"

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        box=box,
        cls=cls,
        weight_decay=weight_decay,
        multi_scale=multi_scale,
        mosaic=1.0,
        copy_paste=0.4,
        scale=0.9,
        translate=0.1,
        degrees=5.0,
        shear=2.0,
        perspective=0.0005,
        close_mosaic=10,
        project=str(REPO_ROOT / "yolo" / "runs"),
        name="yolo11n_fridge",
    )


def test_model(model, dataset_path, conf=0.25, iou=0.6):
    """
    Run object detection and export:
    1) images with YOLO bounding boxes
    2) per-detection crops for downstream DINO classification
    """
    source_path = Path(dataset_path)
    if not source_path.is_absolute():
        source_path = (REPO_ROOT / source_path).resolve()

    output_root = REPO_ROOT / "yolo" / "runs" / "eval"  # clean this up maybe
    crops_root = output_root / "crops"
    if crops_root.exists():
        shutil.rmtree(crops_root)
    crops_root.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(source_path),
        conf=conf,
        iou=iou,
        save=True,
        save_crop=False,
        project=str(output_root),
        name="predictions",
        exist_ok=True,
    )

    pred_ingredients = {}
    crop_manifest = {}

    for image_idx, result in enumerate(results):
        best_pred_per_ingredient = {}
        image_name = Path(result.path).stem if result.path else f"image_{image_idx}"
        image = result.orig_img

        for box_idx, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            ingredient = model.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if ingredient not in best_pred_per_ingredient or confidence > best_pred_per_ingredient[ingredient]:
                best_pred_per_ingredient[ingredient] = confidence

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            crop_path = crops_root / f"{image_name}_{box_idx}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_manifest[crop_path.name] = {
                "source_image": Path(result.path).name if result.path else image_name,
                "box_index": box_idx,
                "yolo_label": ingredient,
                "yolo_confidence": round(confidence, 6),
                "bbox_xyxy": [x1, y1, x2, y2],
            }

        ingredient_list = [
            {"ingredient": cls, "confidence": round(conf, 3)} for cls, conf in best_pred_per_ingredient.items()
        ]
        print(f"{Path(result.path).name}: {ingredient_list}")
        pred_ingredients[f"{Path(result.path).name}"] = ingredient_list

    manifest_path = crops_root / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(crop_manifest, f, indent=2)

    print(f"Prepared crops for DINO under {crops_root}")
    print(f"Wrote crop manifest: {manifest_path}")
    return pred_ingredients


def evaluate_models(model_entries, data_yaml):
    """
    Evaluate multiple YOLO checkpoints and prints a compact comparison table for writeup
    """

    rows = []
    for label, weights_path in model_entries:
        path = Path(weights_path)
        if not path.exists():
            print(f"[SKIP] {label}: checkpoint not found at {path}")
            continue

        model = YOLO(str(path))
        metrics = model.val(data=str(data_yaml), verbose=False, project=str(REPO_ROOT / "yolo" / "runs"))
        row = {
            "label": label,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        rows.append(row)

    print("\nModel comparison (higher better):")
    print(f"{'Label':<35} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50_95':>10}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['label']:<35} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['mAP50']:>10.4f} "
            f"{row['mAP50_95']:>10.4f}"
        )


def inference_sweep(model, data_yaml="ingredients_data/data.yaml"):
    confs = [0.10, 0.25, 0.40, 0.55]
    ious = [0.50, 0.60, 0.70]
    for conf in confs:
        for iou in ious:
            metrics = model.val(data=data_yaml, conf=conf, iou=iou)
            print(f"conf={conf:.2f}, iou={iou:.2f}")
            print(metrics.box)


def parse_json(jsonl_path):
    records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records[Path(row["image"]).name] = row.get("true_ingredients", [])
    return records


def classification_accuracy(pred_ingredients, actual_ingredients):
    """
    Given detected ingredients and ground truth ingredients,
    compute per-image precision/recall/F1 and macro averages.
    """
    per_image = {}
    total_tp = total_fp = total_fn = 0

    for image_name, preds in pred_ingredients.items():
        pred_set = {p["ingredient"].lower() for p in preds}
        print(f"Predicted set: {pred_set}")
        true_set = set(i.lower() for i in actual_ingredients.get(image_name, []))
        print(f"True set: {true_set}")

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_image[image_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if (global_precision + global_recall)
        else 0.0
    )

    summary = {
        "global_precision": round(global_precision, 4),
        "global_recall": round(global_recall, 4),
        "global_f1": round(global_f1, 4),
        "images_evaluated": len(per_image),
    }

    return {"summary": summary, "per_image": per_image}

def main():
    data_yaml = REPO_ROOT / "ingredients_data" / "data.yaml"

    # Training (already complete)
    train_model(
        epochs=30,
        imgsz=896,
        batch=16, 
        patience=25,
        lr0=0.001,
        lrf=0.01,
        box=8.0,
        cls=0.5,
        weight_decay=0.0005,
        workers=2,
    ) 

    # Compare model size + tuning effects under identical validation settings.
    model_entries = [
        ("yolo11n baseline", REPO_ROOT / "yolo_runs" / "ingredients_yolo11n" / "weights" / "best.pt"),
        ("yolo11n baseline (run 2)", REPO_ROOT / "yolo_runs" / "ingredients_yolo11n-2" / "weights" / "best.pt"),
        ("yolo11s baseline", REPO_ROOT / "yolo_runs" / "ingredients_yolo11s" / "weights" / "best.pt"),
        ("yolo11n tuned", REPO_ROOT / "yolo_runs" / "ingredients_tune11n" / "weights" / "best.pt"),
    ]
    # evaluate_models(model_entries, data_yaml)

    # Qualitative inspection on a selected checkpoint:
    # print("\n Qualitative Metrics")
    # best_checkpoint = REPO_ROOT / "yolo_runs" / "ingredients_tune11n" / "weights" / "best.pt" #copy of weights from best epoch
    # pred_ingredients = test_model(YOLO(str(best_checkpoint)), dataset_path=REPO_ROOT / "eval_data" / "images")
    
    # actual_ingredients = parse_json(REPO_ROOT/"eval_data"/ "labels.jsonl")
    # classification_accuracy(pred_ingredients, actual_ingredients)
   
    # Optional evaluation sweep:
    # inference_sweep(model, data_yaml=str(base_dir / "ingredients_data" / "data.yaml"))


if __name__ == "__main__":
    main()
