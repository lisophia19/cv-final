from inference_sdk import InferenceHTTPClient
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
IMAGE_PATH = "./data/fridge_test2.jpeg"   # change if needed

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

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