from pathlib import Path
from ultralytics import YOLO


def train_model(tune=False):
    #Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
    model = YOLO("yolo11n.pt")

    #Start training on your custom dataset
    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "ingredients_data" / "data.yaml"

    if tune:
        model.tune(
            data=str(data_yaml),
            epochs=50,  # epochs per trial
            iterations=10,  # number of tuning trials
            imgsz=800,
            batch=16,
            optimizer="AdamW",
            project=str(base_dir / "runs"),
            name="ingredients_tune11n",
        )
    else:
        model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=800,
            batch=16,
            patience=25,
            optimizer="AdamW",
            project=str(base_dir / "runs"),
            name="ingredients_yolo11s",
        )

# More complex training function with finetuning and specific hyperparam definition
def train_model2():
    # Editable training params (DO THIS FIRST)
    train_cfg = {
        "model": "yolo11s.pt",
        "epochs": 50,
        "imgsz": 700,
        "batch": 16,
        "patience": 25,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "box": 7.5,
        "cls": 0.5,
        "weight_decay": 0.0005,
    }

    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "ingredients_data" / "data.yaml"
    model = YOLO(train_cfg["model"])

    if not tune:
        model.train(
            data=str(data_yaml),
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            batch=train_cfg["batch"],
            patience=train_cfg["patience"],
            optimizer=train_cfg["optimizer"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            box=train_cfg["box"],
            cls=train_cfg["cls"],
            weight_decay=train_cfg["weight_decay"],
            project=str(base_dir / "runs"),
            name="yolo11s_params",
        )


def test_model(model, dataset_path, output_dir="runs/dino_input", conf=0.25, iou=0.6):
    """
    Run object detection and export:
    1) images with YOLO bounding boxes 
    2) per-detection crops for downstream DINO classification
    """
    base_dir = Path(__file__).resolve().parent
    source_path = Path(dataset_path)
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = (base_dir / output_root).resolve()

    crops_root = output_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(source_path),
        conf=conf,
        iou=iou,
        save=True,
        save_crop=True,
        project=str(output_root),
        name="predictions",
        exist_ok=True,
    )

    detections_for_dino = []

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

            detections_for_dino.append(
                {
                    "image_name": image_name,
                    "ingredient": ingredient,
                    "confidence": round(confidence, 3),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_array": crop,
                    "crop_index": box_idx,
                }
            )

        ingredient_list = [
            {"ingredient": cls, "confidence": round(conf, 3)} for cls, conf in best_pred_per_ingredient.items()
        ]
        print(f"{Path(result.path).name}: {ingredient_list}")

    print(f"Prepared {len(detections_for_dino)} crops for DINO under {output_root / 'predictions'}")
    return detections_for_dino


def inference_sweep(model, data_yaml="ingredients_data/data.yaml"):
    confs = [0.10, 0.25, 0.40, 0.55]
    ious = [0.50, 0.60, 0.70]
    for conf in confs:
        for iou in ious:
            metrics = model.val(data=data_yaml, conf=conf, iou=iou)
            print(f"conf={conf:.2f}, iou={iou:.2f}")
            print(metrics.box)


def main():
    base_dir = Path(__file__).resolve().parent

    # Inference / crop export path for DINO integration.
    # model = YOLO(str(base_dir / "runs" / "ingredients_tune11n" / "weights" / "best.pt"))
    model = YOLO(str(base_dir / "runs" / "ingredients_yolo11n" / "weights" / "best.pt"))

    test_model(model, dataset_path="fridge_data")

    # Optional evaluation sweep:
    # inference_sweep(model, data_yaml=str(base_dir / "ingredients_data" / "data.yaml"))


if __name__ == "__main__":
    main()
