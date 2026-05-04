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
CONFIDENCE_THRESHOLD = 0.5  

INGREDIENT_EMOJI: dict[str, str] = {
    "apple": "🍎", "banana": "🍌", "orange": "🍊", "lemon": "🍋", "lime": "🍋",
    "pear": "🍐", "strawberry": "🍓", "watermelon": "🍉", "avocado": "🥑",
    "papaya": "🥭", "jackfruit": "🥭", "hog plum": "🍑",
    "tomato": "🍅", "tree tomato": "🍅", "eggplant": "🍆", "green eggplant": "🍆",
    "bell pepper": "🫑", "chili pepper": "🌶️", "chili powder": "🌶️",
    "carrot": "🥕", "corn": "🌽", "broccoli": "🥦", "cauliflower": "🥦",
    "cucumber": "🥒", "onion": "🧅", "onion leaves": "🌿", "garlic": "🧄",
    "ginger": "🫚", "mushroom": "🍄", "potato": "🥔", "sweet potato": "🍠",
    "cabbage": "🥬", "spinach": "🥬", "indian spinach": "🥬", "kimchi": "🥬",
    "moringa leaves": "🌿", "taro leaves": "🌿", "garden cress": "🌿",
    "fiddlehead ferns": "🌿", "stinging nettle": "🌿", "green mint": "🌿",
    "coriander": "🌿", "asparagus": "🌿", "artichoke": "🌿", "okra": "🌿",
    "seaweed": "🌿", "bamboo shoots": "🌿",
    "bread": "🍞", "cheese": "🧀", "paneer": "🧀", "egg": "🥚",
    "bacon": "🥓", "ham": "🍖", "sausage": "🌭",
    "chicken": "🍗", "chicken gizzards": "🍗",
    "beef": "🥩", "pork": "🥩", "mutton": "🥩",
    "minced meat": "🥩", "buffalo meat": "🥩",
    "fish": "🐟", "crab meat": "🦀",
    "rice": "🍚", "beaten rice": "🍚", "noodles": "🍜", "cornflakes": "🥣",
    "milk": "🥛", "butter": "🧈", "olive oil": "🫒",
    "salt": "🧂", "sugar": "🍬", "ice": "🧊",
    "soy sauce": "🥢", "ketchup": "🥫", "mayonnaise": "🥫",
    "beans": "🫘", "red beans": "🫘", "black beans": "🫘", "broad beans": "🫘",
    "long beans": "🫘", "soybean": "🫘", "soy chunks": "🫘",
    "green soybean": "🫘", "chickpeas": "🫘",
    "black lentils": "🫘", "green lentils": "🫘",
    "red lentils": "🫘", "yellow lentils": "🫘",
    "green peas": "🫛", "garden peas": "🫛", "pea": "🫛",
    "tofu": "🍱", "pumpkin": "🎃", "radish": "🌱", "turnip": "🌱",
    "beetroot": "🟣", "cinnamon": "🟫", "walnut": "🌰", "wheat": "🌾",
    "ash gourd": "🥒", "bitter gourd": "🥒", "bottle gourd": "🥒",
    "pointed gourd": "🥒", "snake gourd": "🥒", "sponge gourd": "🥒",
    "chayote": "🥒", "cassava": "🥔", "taro root": "🥔",
    "moringa drumsticks": "🌿",
}


def confidence_tier(conf: float) -> str:
    if conf >= 0.8:
        return "🟢"
    if conf >= 0.5:
        return "🟡"
    return "🔴"


def chip(text: str, bg: str, fg: str = "#1a1a1a") -> str:
    return (
        f'<span style="background-color:{bg};color:{fg};'
        f'padding:3px 10px;border-radius:12px;margin:2px 4px 2px 0;'
        f'display:inline-block;font-size:0.9em">{text}</span>'
    )

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
    """Loads DINO detector and DINOv2 + LoRA classifier. Cached to avoid reloading on every run."""
    detector = load_detector("cpu")
    backbone, classifier, processor, class_names = load_classifier("cpu")
    return detector, backbone, classifier, processor, class_names


@st.cache_resource
def load_recipe_index():
    if not SAMPLE_RECIPE_PATH.exists():
        st.error(
            "Recipe sample file not found at "
            f"{SAMPLE_RECIPE_PATH}. Add a recipe file before running retrieval."
        )
        st.stop()
    return build_index_from_paths([SAMPLE_RECIPE_PATH])


def detect_ingredients(models, image: Image.Image) -> tuple[list[Ingredient], Image.Image]:
    """Runs the DINO pipeline. Returns deduped ingredient list and annotated image."""
    detector, backbone, classifier, processor, class_names = models
    detections, _ = detect_and_classify(
        image, detector, backbone, classifier, processor, class_names,
        method="tokencut", device="cpu",
    )

    # Filters low-confidence and dedups by max confidence per ingredient 
    # in order to keep the best box 
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

    with st.expander("How this works"):
        st.markdown(
            "**📷 Photo** → **🔍 DINOv2 + TokenCut** finds object regions → "
            "**🏷️ DINOv2 + LoRA classifier** labels each region → "
            "**🍔 Recipe retrieval** ranks recipes by ingredient overlap."
        )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "chosen_sample" not in st.session_state:
        st.session_state.chosen_sample = None

    sample_paths = sorted((BASE_DIR / "fridge_data").glob("fridge_test*.jpg"))
    if sample_paths:
        st.markdown("**Try a sample fridge:**")
        cols = st.columns(len(sample_paths))
        for i, (col, path) in enumerate(zip(cols, sample_paths), 1):
            with col:
                if st.button(f"Test fridge {i}", key=f"sample_btn_{i}", width="stretch"):
                    st.session_state.chosen_sample = str(path)
                    st.session_state.uploader_key += 1

    uploaded = st.file_uploader(
        "Or upload your own image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded is not None:
        st.session_state.chosen_sample = None
        image = Image.open(uploaded).convert("RGB")
    elif st.session_state.chosen_sample:
        image = Image.open(st.session_state.chosen_sample).convert("RGB")
    else:
        st.info("Pick a sample fridge above or upload your own image.")
        return

    with st.spinner("Loading DINO models (one-time)..."):
        models = load_models()

    with st.spinner("Detecting ingredients (DINO pipeline, ~30s on CPU)..."):
        ingredients, annotated = detect_ingredients(models, image)

    left, right = st.columns(2)
    with left:
        st.subheader("Detections")
        st.image(annotated, width="stretch")
    with right:
        st.subheader("Ingredients")
        if not ingredients:
            st.warning("No high-confidence ingredients detected.")
        else:
            st.caption("🟢 high confidence  ·  🟡 medium  ·  🔴 low")
            for ing in ingredients:
                name = ing["ingredient"]
                conf = ing["confidence"]
                emoji = INGREDIENT_EMOJI.get(name.lower(), "🥗")
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"{confidence_tier(conf)} {emoji} **{name}**")
                with c2:
                    st.progress(conf, text=f"{conf:.2f}")

    st.subheader("Suggested Recipes")
    recipes = get_recipes(ingredients)
    if not recipes:
        st.write("No recipes to suggest.")
    else:
        for r in recipes:
            with st.container(border=True):
                n_matched = len(r["matched_ingredients"])
                n_total = n_matched + len(r["missing_ingredients"])
                title_col, metric_col = st.columns([3, 1])
                with title_col:
                    st.markdown(f"### {r['title']}")
                with metric_col:
                    st.metric("Match", f"{n_matched}/{n_total}")
                if r["matched_ingredients"]:
                    chips = "".join(chip(x, "#a8e6c1") for x in r["matched_ingredients"])
                    st.markdown(f"**You have:** {chips}", unsafe_allow_html=True)
                if r["missing_ingredients"]:
                    chips = "".join(chip(x, "#e9ecef") for x in r["missing_ingredients"])
                    st.markdown(f"**You need:** {chips}", unsafe_allow_html=True)
                st.caption(f"Score: {r['score']}")


if __name__ == "__main__":
    main()
