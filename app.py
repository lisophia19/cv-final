"""
Streamlit demo for the vision recipe recommender pipeline.

Run with:
    streamlit run app.py
from inside the cv-final directory.
"""

from pathlib import Path
from typing import TypedDict

import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "ingredients_yolo11n" / "weights" / "best.pt"
CONF_THRESHOLD = 0.25


class Ingredient(TypedDict):
    ingredient: str
    confidence: float


class Recipe(TypedDict):
    title: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    score: float


@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        st.error(f"Model weights not found at {MODEL_PATH}. Train the YOLO model first.")
        st.stop()
    return YOLO(str(MODEL_PATH))


def detect_ingredients(model: YOLO, image: Image.Image) -> tuple[list[Ingredient], Image.Image]:
    """Run YOLO detection and return deduped ingredient list + annotated image."""
    results = model.predict(source=image, conf=CONF_THRESHOLD, save=False, verbose=False)
    result = results[0]

    best_per_ingredient: dict[str, float] = {}
    for box in result.boxes:
        class_id = int(box.cls.item())
        name = model.names[class_id]
        confidence = float(box.conf.item())
        if name not in best_per_ingredient or confidence > best_per_ingredient[name]:
            best_per_ingredient[name] = confidence

    ingredients: list[Ingredient] = [
        {"ingredient": name, "confidence": round(conf, 3)}
        for name, conf in sorted(best_per_ingredient.items(), key=lambda kv: -kv[1])
    ]

    annotated_array = result.plot()  # BGR numpy array with boxes drawn
    annotated_image = Image.fromarray(annotated_array[..., ::-1])  # BGR -> RGB
    return ingredients, annotated_image


def get_recipes(ingredients: list[Ingredient]) -> list[Recipe]:
    """
    TODO(member-2): replace with real retrieval against Recipe1M+.

    Expected contract:
      - input: list of {ingredient: str, confidence: float}
      - output: ranked list of recipes, best first
    """
    if not ingredients:
        return []
    detected_names = [i["ingredient"] for i in ingredients]
    return [
        {
            "title": "[STUB] Recipe retrieval not yet wired in",
            "matched_ingredients": detected_names,
            "missing_ingredients": [],
            "score": 0.0,
        }
    ]


def main() -> None:
    st.set_page_config(page_title="Recipe Recommender", layout="wide")
    st.title("Vision Recipe Recommender")
    st.caption("Upload a kitchen or fridge photo. The system detects ingredients and suggests recipes.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Waiting for an image...")
        return

    image = Image.open(uploaded).convert("RGB")
    model = load_model()

    with st.spinner("Detecting ingredients..."):
        ingredients, annotated = detect_ingredients(model, image)

    left, right = st.columns(2)
    with left:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)
    with right:
        st.subheader("Ingredients")
        if not ingredients:
            st.warning("No ingredients detected above the confidence threshold.")
        else:
            st.table(ingredients)

    st.subheader("Suggested Recipes")
    recipes = get_recipes(ingredients)
    if not recipes:
        st.write("No recipes to suggest.")
    else:
        for r in recipes:
            st.markdown(f"**{r['title']}**  \nScore: {r['score']}  \nMatched: {', '.join(r['matched_ingredients']) or '—'}")


if __name__ == "__main__":
    main()
