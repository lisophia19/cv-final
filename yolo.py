from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n.pt")

# Start training on your custom dataset
base_dir = Path(__file__).resolve().parent
data_yaml = base_dir / "ingredients_data" / "data.yaml"

model.train(
    data=str(data_yaml),
    epochs=30,
    imgsz=640,
    workers=0,
    project=str(base_dir / "runs"),
    name="ingredients_yolo11n",
)