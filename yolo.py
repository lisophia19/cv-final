from ultralytics import YOLO
from pathlib import Path

def train_model(tune=False):
    #load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
    model = YOLO("yolo11n.pt")

    # start training on your custom dataset
    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "ingredients_data" / "data.yaml"

    if tuning:
        model.tune(
            data=str(data_yaml),
            epochs=50,      # epochs per trial
            iterations=10,  # number of tuning trials
            imgsz=800,
            batch=16,
            optimizer="AdamW",
            project=str(base_dir / "runs"),
            name="ingredients_tune11n"
        )
    else:
        model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=800,
            batch=16,
            patience = 25,
            optimizer="AdamW",
            project=str(base_dir / "runs"),
            name="ingredients_yolo11s",
        )

#more complex training function with finetuning and specific hyperparam definition
def train_model2():
    # editable training params (DO THIS FIRST)
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
        "cls": 0.5
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
            box=train_cfg["box"]
            cls=train_cfg["cls"],
            weight_decay=train_cfg["weight_decay"],
            name="yolo11s_params,
        )

#test model on fridge images with a low confidence threshold
def test_model(model, dataset_path):
    results = model.predict(
        source=dataset_path,
        conf = 0.25,
        save = True
    )

    for r in results:
        best_pred_ingredient = {}

        for box in r.boxes:
            class_id = int(box.cls.item())
            ingredient = model.names[class_id]
            confidence = float(box.conf.item())

            if ingredient not in best_pred_ingredient or confidence > best_pred_ingredient[ingredient]:
                best_pred_ingredient[ingredient] = confidence

        ingredient_list = [
            {"ingredient": cls, "confidence": round(conf, 3)} for cls, conf in best_pred_ingredient.items()
        ]
        print(f"{Path(r.path).name}: {ingredient_list}")
    
def inference_sweep(model):
    confs = [0.10, 0.25, 0.40, 0.55] #min needed to keep prediction
    ious  = [0.50, 0.60, 0.70] # how aggresively overlapping boxes are supressed
    for conf in confs:
        for iou in ious:
            metrics = model.val(
                data="ingredients_data/data.yaml",
                conf=conf,
                iou=iou
            )
            print(f"conf={conf:.2f}, iou={iou:.2f}")

def main():
    #training
    train_model(tune=True)

    #testing/evaluation
    model = YOLO("runs/ingredients_tune11n/weights/best.pt")
    test_model(model, "./fridge_data")
    inference_sweep()

if __name__ == '__main__':
    main()
