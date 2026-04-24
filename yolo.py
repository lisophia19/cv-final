from ultralytics import YOLO
from pathlib import Path

def train_model(tuning=False):
    #load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
    model = YOLO("yolo11.pt")

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
            name="ingredients_tune"
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
            name="ingredients_yolo11n",
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

def main():
    #train_model(tuning=True)

    model = YOLO("runs/ingredients_yolo11n/weights/best.pt")
    test_model(model, "./fridge_data")

if __name__ == '__main__':
    main()




#more complex training function with finetuning and specific hyperparam definition
def train_model2(tuning=False):
    # editable training params
    train_cfg = {
        "model": "yolo11s.pt",
        "epochs": 80,
        "imgsz": 800,
        "batch": 16,
        "patience": 25,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
    }

    # if we wna to utilize auto tuning
    tune_cfg = {
        "epochs": 40,
        "iterations": 10,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
    }

    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "ingredients_data" / "data.yaml"
    model = YOLO(train_cfg["model"])

    if tuning:
        model.tune(
            data=str(data_yaml),
            epochs=tune_cfg["epochs"],
            iterations=tune_cfg["iterations"],
            imgsz=tune_cfg["imgsz"],
            batch=tune_cfg["batch"],
            optimizer=tune_cfg["optimizer"],
            project=str(base_dir / "runs"),
            name="ingredients_tune",
        )
    else:
        model.train(
            data=str(data_yaml),
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            batch=train_cfg["batch"],
            patience=train_cfg["patience"],
            optimizer=train_cfg["optimizer"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            weight_decay=train_cfg["weight_decay"],
            name="ingredients_yolo11s_knobs",
        )