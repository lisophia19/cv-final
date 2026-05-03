"""
Streamlit demo for the vision recipe recommender pipeline.

Uses the full DINO pipeline (TokenCut detection + DINOv2 + LoRA classification)
plus Martin's recipe retrieval module.

Run with:
    streamlit run app.py
from inside the cv-final directory.
"""

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dino_detect import load_model as load_detector
from dino_pipeline import detect_and_classify, load_classifier
from recipe_retrieval.integrate import retrieve_with_reconciled_vocab
from recipe_retrieval.pipeline import build_index_from_paths
from recipe_retrieval.text import ingredient_key

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RECIPE_PATH = BASE_DIR / "fridge_data" / "sample_recipes.jsonl"
TEAM_ALIAS_PATH = BASE_DIR / "fridge_data" / "team_ingredient_alias.json"
CONFIDENCE_THRESHOLD = 0.5  # drop classifier predictions below this


class Ingredient(TypedDict):
    ingredient: str
    confidence: float


class Recipe(TypedDict):
    title: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    score: float


@st.cache_resource
def load_models():
    """Load DINO detector and DINOv2 + LoRA classifier (cached)."""
    detector = load_detector("cpu")
    backbone, classifier, processor, class_names = load_classifier("cpu")
    return detector, backbone, classifier, processor, class_names


@st.cache_resource
def load_recipe_index():
    if not SAMPLE_RECIPE_PATH.exists():
        st.error(
            "Recipe sample file not found at "
            f"{SAMPLE_RECIPE_PATH}. Add a recipe corpus file before running retrieval."
        )
        st.stop()
    return build_index_from_paths([SAMPLE_RECIPE_PATH])


def detect_ingredients(models, image: Image.Image) -> tuple[list[Ingredient], Image.Image]:
    """Run the DINO pipeline. Returns deduped ingredient list + annotated image."""
    detector, backbone, classifier, processor, class_names = models
    detections, _ = detect_and_classify(
        image, detector, backbone, classifier, processor, class_names,
        method="tokencut", device="cpu",
    )

    # Filter low-confidence; dedup by max confidence per ingredient (keep best box)
    best_conf: dict[str, float] = {}
    best_box: dict[str, tuple[int, int, int, int]] = {}
    for d in detections:
        if d.confidence < CONFIDENCE_THRESHOLD:
            continue
        if d.ingredient not in best_conf or d.confidence > best_conf[d.ingredient]:
            best_conf[d.ingredient] = d.confidence
            best_box[d.ingredient] = d.box

    ingredients: list[Ingredient] = [
        {"ingredient": name, "confidence": round(conf, 3)}
        for name, conf in sorted(best_conf.items(), key=lambda kv: -kv[1])
    ]

    img_np = np.array(image).copy()
    for name, box in best_box.items():
        x, y, w, h = box
        label = f"{name} {best_conf[name]:.2f}"
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_np, label, (x, max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    annotated_image = Image.fromarray(img_np)
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

    with st.spinner("Loading DINO models (one-time)..."):
        models = load_models()

    with st.spinner("Detecting ingredients (DINO pipeline, ~30s on CPU)..."):
        ingredients, annotated = detect_ingredients(models, image)

    left, right = st.columns(2)
    with left:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)
    with right:
        st.subheader("Ingredients")
        if not ingredients:
            st.warning("No high-confidence ingredients detected.")
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
