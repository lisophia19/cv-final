from __future__ import annotations

import ast
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATASET = BASE_DIR / "FOOD-INGREDIENTS dataset.v2i.yolov11"
TARGET_DATASET = BASE_DIR / "FOOD-INGREDIENTS dataset.v2i.yolov11_reduced"


def parse_data_yaml(data_yaml: Path) -> tuple[list[str], dict[str, str]]:
    lines = data_yaml.read_text().splitlines()
    names_line = next((line for line in lines if line.startswith("names:")), None)
    if names_line is None:
        raise ValueError(f"Could not find names list in {data_yaml}")
    names = ast.literal_eval(names_line.split("names:", 1)[1].strip())

    split_paths: dict[str, str] = {}
    for key in ("train", "val", "test"):
        match = next((line for line in lines if line.startswith(f"{key}:")), None)
        if match:
            split_paths[key] = match.split(":", 1)[1].strip()
    return names, split_paths


def reduced_class(name: str) -> str:
    n = name.lower().strip()
    if n == "na":
        return "unknown"

    # Protein classes
    if any(k in n for k in ("beef", "buffalo", "chicken", "mutton", "pork", "ham", "sausage", "meat", "gizzards")):
        return "meat"
    if any(k in n for k in ("fish", "crab", "seaweed")):
        return "seafood"
    if any(k in n for k in ("milk", "paneer", "cheese", "butter", "tofu")):
        return "dairy_tofu"
    if any(k in n for k in ("lentil", "bean", "pea", "chickpea", "soy", "daal")):
        return "legume"
    if "egg" == n:
        return "egg"

    # Produce classes
    if any(k in n for k in ("spinach", "saag", "mint", "leaves", "nett", "fern", "cress", "gundruk")):
        return "leafy_green"
    if any(
        k in n
        for k in (
            "apple",
            "banana",
            "avocado",
            "jackfruit",
            "lemon",
            "lime",
            "orange",
            "papaya",
            "pear",
            "strawberry",
            "watermelon",
            "tree tomato",
            "hog plum",
            "lapsi",
        )
    ):
        return "fruit"
    if any(
        k in n
        for k in (
            "gourd",
            "eggplant",
            "brinjal",
            "okra",
            "potato",
            "carrot",
            "radish",
            "cabbage",
            "cauliflower",
            "broccoli",
            "cucumber",
            "onion",
            "garlic",
            "ginger",
            "tomato",
            "corn",
            "cassava",
            "mushroom",
            "turnip",
            "taro",
            "bamboo shoots",
            "asparagus",
            "beetroot",
            "chayote",
            "pointed gourd",
            "long beans",
            "sweet potato",
            "moringa",
            "chili pepper",
            "artichoke",
            "pumpkin",
        )
    ):
        return "vegetable"

    # Pantry classes
    if any(k in n for k in ("rice", "wheat", "bread", "cornflakes", "beaten rice")):
        return "grain_starch"
    if any(k in n for k in ("noodle", "thukpa", "chowmein")):
        return "noodle"
    if any(k in n for k in ("chili powder", "cinnamon", "salt", "sugar")):
        return "seasoning"
    if any(k in n for k in ("soy sauce", "ketchup", "mayonnaise", "kimchi")):
        return "condiment"
    if "olive oil" in n:
        return "oil"
    if any(k in n for k in ("walnut", "peanut", "nut")):
        return "nut"
    if n == "ice":
        return "other"
    return "other"


def rewrite_label_file(src: Path, dst: Path, old_to_new_id: dict[int, int]) -> None:
    if not src.exists():
        dst.write_text("")
        return
    out_lines: list[str] = []
    for raw in src.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        new_id = old_to_new_id[old_id]
        parts[0] = str(new_id)
        out_lines.append(" ".join(parts))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def main() -> None:
    source_yaml = SOURCE_DATASET / "data.yaml"
    names, _ = parse_data_yaml(source_yaml)

    reduced_names: list[str] = []
    reduced_to_id: dict[str, int] = {}
    old_to_new_id: dict[int, int] = {}
    for old_id, name in enumerate(names):
        reduced = reduced_class(name)
        if reduced not in reduced_to_id:
            reduced_to_id[reduced] = len(reduced_names)
            reduced_names.append(reduced)
        old_to_new_id[old_id] = reduced_to_id[reduced]

    if TARGET_DATASET.exists():
        shutil.rmtree(TARGET_DATASET)
    TARGET_DATASET.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src_images = SOURCE_DATASET / split / "images"
        src_labels = SOURCE_DATASET / split / "labels"
        dst_images = TARGET_DATASET / split / "images"
        dst_labels = TARGET_DATASET / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Symlink images to avoid duplicating large data.
        for image_file in src_images.iterdir():
            target = dst_images / image_file.name
            target.symlink_to(image_file.resolve())

        for src_label in src_labels.iterdir():
            dst_label = dst_labels / src_label.name
            rewrite_label_file(src_label, dst_label, old_to_new_id)

    yaml_out = TARGET_DATASET / "data.yaml"
    yaml_out.write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                "",
                f"nc: {len(reduced_names)}",
                f"names: {reduced_names}",
                "",
                "# Generated from FOOD-INGREDIENTS dataset.v2i.yolov11",
            ]
        )
        + "\n"
    )

    print(f"Reduced classes: {len(reduced_names)}")
    for idx, name in enumerate(reduced_names):
        print(f"{idx}: {name}")
    print(f"Wrote remapped dataset to: {TARGET_DATASET}")


if __name__ == "__main__":
    main()
