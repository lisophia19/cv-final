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
from recipe_retrieval.integrate import retrieve_with_reconciled_vocab
from recipe_retrieval.pipeline import build_index_from_paths
from recipe_retrieval.text import ingredient_key
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "ingredients_yolo11n" / "weights" / "best.pt"
SAMPLE_RECIPE_PATH = BASE_DIR / "fridge_data" / "sample_recipes.jsonl"
TEAM_ALIAS_PATH = BASE_DIR / "fridge_data" / "team_ingredient_alias.json"
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


@st.cache_resource
def load_recipe_index():
    if not SAMPLE_RECIPE_PATH.exists():
        st.error(
            "Recipe sample file not found at "
            f"{SAMPLE_RECIPE_PATH}. Add a recipe corpus file before running retrieval."
        )
        st.stop()
    return build_index_from_paths([SAMPLE_RECIPE_PATH])


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
    if not ingredients:
        return []
    index = load_recipe_index()
    result = retrieve_with_reconciled_vocab(
        ingredients,
        index=index,
        ranker="penalty_aware",
        k=5,
        alias_path=TEAM_ALIAS_PATH if TEAM_ALIAS_PATH.exists() else None,
    )
    detected_keys = {
        ingredient_key(item.canonical)
        for item in result.normalized
        if ingredient_key(item.canonical)
    }
    out: list[Recipe] = []
    for ranked in result.top_k:
        rec = index.recipes.get(ranked.recipe_id)
        if rec is None:
            continue
        recipe_keys = {ingredient_key(x) for x in rec.ingredients if ingredient_key(x)}
        matched_keys = sorted(recipe_keys & detected_keys)
        missing_keys = sorted(recipe_keys - detected_keys)
        out.append(
            {
                "title": rec.title,
                "matched_ingredients": matched_keys,
                "missing_ingredients": missing_keys,
                "score": round(ranked.score, 3),
            }
        )
    return out


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
            st.markdown(
                f"**{r['title']}**  \n"
                f"Score: {r['score']}  \n"
                f"Matched: {', '.join(r['matched_ingredients']) or '—'}  \n"
                f"Missing: {', '.join(r['missing_ingredients']) or '—'}"
            )


if __name__ == "__main__":
    main()
