from inference_sdk import InferenceHTTPClient
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
IMAGE_PATH = "./data/fridge_test2.jpg"   # change if needed

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

client.configure(InferenceConfiguration(confidence_threshold=0.1))
result = client.infer(IMAGE_PATH, model_id="food-ingredients-dataset/3")
print(result)

predictions = result.get("predictions", [])
best_pred_ingredient = {}

for pred in predictions:
    ingredient = pred["class"]
    confidence = pred["confidence"]

    if ingredient not in best_pred_ingredient or confidence > best_pred_ingredient[ingredient]:
        best_pred_ingredient[ingredient] = confidence

ingredient_list = [
    {"ingredient": cls, "confidence": round(conf, 3)}
    for cls, conf in best_pred_ingredient.items()
]

print(ingredient_list)

# python test.py
# {'inference_id': '995fa942-c865-48b4-a492-9034ee5a4674', 'time': 0.007789901923388243, 'image': {'width': 512, 'height': 269}, 'predictions': [{'x': 180.5, 'y': 203.5, 'width': 53.0, 'height': 51.0, 'confidence': 0.7593904137611389, 'class': 'Akabare Khursani', 'class_id': 0, 'detection_id': '97bfad1c-ec59-4159-9dab-c3e361e34331'}]}
# [{'ingredient': 'Akabare Khursani', 'confidence': 0.759}]