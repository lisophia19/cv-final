"""
Full DINO-only detection and classification pipeline.

Combines:
  - dino_detect.py: TokenCut/threshold box generation 
  - dino/lora/: DINOv2 and LoRA fine-tuned classifier 

Pipeline:
    image -> dino_detect (boxes) -> crop each box -> classifier -> ingredient and confidence

Usage:
    python dino_pipeline.py fridge_data/fridge_test1.jpg
    python dino_pipeline.py <image> --method tokencut --tau 0.15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModel

from dino_detect import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOKENCUT_MAX_OBJECTS,
    DEFAULT_TOKENCUT_TAU,
    MAX_BOX_AREA_FRACTION,
    MIN_BOX_AREA_FRACTION,
    attention_to_boxes,
    get_attention_map,
    get_patch_features,
    load_model as load_detector,
    preprocess as preprocess_for_detect,
    tokencut_to_boxes,
)

BASE_DIR = Path(__file__).resolve().parent
LORA_ADAPTER_DIR = BASE_DIR / "dino" / "lora" / "dinov2_lora_adapters"
CLASSIFIER_PATH = BASE_DIR / "dino" / "lora" / "dinov2_lora_classifier.pt"
CLASS_NAMES_PATH = BASE_DIR / "ingredients_data" / "data.yaml"
DEFAULT_OUT_DIR = BASE_DIR / "runs" / "dino_pipeline"
MODEL_NAME = "facebook/dinov2-base"


@dataclass
class Detection:
    box: tuple[int, int, int, int]   # (x, y, w, h) original image coordinates
    ingredient: str
    confidence: float


def load_classifier(device: str = "cpu"):
    """Load Nina's DINOv2, LoRA, and linear-classifier stack."""
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)
    backbone = PeftModel.from_pretrained(backbone, str(LORA_ADAPTER_DIR)).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    with open(CLASS_NAMES_PATH) as f:
        class_names = yaml.safe_load(f)["names"]
    num_classes = len(class_names)

    embed_dim = backbone.config.hidden_size
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(str(CLASSIFIER_PATH), map_location=device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return backbone, classifier, processor, class_names


def classify_crop(
    crop: Image.Image,
    backbone,
    classifier: nn.Module,
    processor,
    class_names: list[str],
    device: str = "cpu",
) -> tuple[str, float]:
    """Classify one image crop. Returns (ingredient, confidence)."""
    pixel_values = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        emb = backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        logits = classifier(emb)
        probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = int(probs.argmax().item())
    return class_names[pred_idx], float(probs[pred_idx].item())


def detect_and_classify(
    image: Image.Image,
    detector,
    backbone,
    classifier: nn.Module,
    processor,
    class_names: list[str],
    method: str = "tokencut",
    threshold: float = DEFAULT_THRESHOLD,
    tau: float = DEFAULT_TOKENCUT_TAU,
    max_objects: int = DEFAULT_TOKENCUT_MAX_OBJECTS,
    max_area_fraction: float = MAX_BOX_AREA_FRACTION,
    layer: int = -1,
    device: str = "cpu",
) -> tuple[list[Detection], np.ndarray]:
    """Full pipeline: image -> list[Detection] and a 2D heatmap/mask for visualization."""
    H, W = image.size[1], image.size[0]
    image_tensor = preprocess_for_detect(image).to(device)

    if method == "threshold":
        attn_map = get_attention_map(detector, image_tensor, layer=layer)
        boxes, viz_map = attention_to_boxes(
            attn_map, (H, W),
            quantile_threshold=threshold,
            min_area_fraction=MIN_BOX_AREA_FRACTION,
            max_area_fraction=max_area_fraction,
        )
    else:  # tokencut
        features = get_patch_features(detector, image_tensor)
        n_patches = features.shape[0]
        grid_size = int(np.sqrt(n_patches))
        assert grid_size * grid_size == n_patches, f"Non-square patch grid: {n_patches}"
        boxes, fg_grid = tokencut_to_boxes(
            features, (H, W),
            grid_size=grid_size,
            tau=tau,
            min_area_fraction=MIN_BOX_AREA_FRACTION,
            max_area_fraction=max_area_fraction,
            max_objects=max_objects,
        )
        viz_map = cv2.resize(fg_grid, (W, H), interpolation=cv2.INTER_NEAREST)

    detections: list[Detection] = []
    for box in boxes:
        x, y, w, h = box
        crop = image.crop((x, y, x + w, y + h))
        ingredient, confidence = classify_crop(
            crop, backbone, classifier, processor, class_names, device
        )
        detections.append(Detection(box=box, ingredient=ingredient, confidence=confidence))

    return detections, viz_map


def visualize_pipeline(
    image: Image.Image,
    detections: list[Detection],
    viz_map: np.ndarray,
    out_path: Path,
):
    """3-panel: original | heatmap overlay | labeled boxes (ingredient and confidence)."""
    img_np = np.array(image)

    attn_norm = (viz_map - viz_map.min()) / (viz_map.max() - viz_map.min() + 1e-8)
    heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap_rgb).astype(np.uint8)

    boxed = img_np.copy()
    for d in detections:
        x, y, w, h = d.box
        label = f"{d.ingredient} {d.confidence:.2f}"
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(boxed, label, (x, max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined = np.concatenate([img_np, overlay, boxed], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full DINO detection and classification pipeline")
    parser.add_argument("image", type=Path)
    parser.add_argument("--method", default="tokencut", choices=["threshold", "tokencut"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tau", type=float, default=DEFAULT_TOKENCUT_TAU)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_TOKENCUT_MAX_OBJECTS)
    parser.add_argument("--max-area-fraction", type=float, default=MAX_BOX_AREA_FRACTION)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not LORA_ADAPTER_DIR.exists():
        raise SystemExit(f"LoRA adapter not found: {LORA_ADAPTER_DIR}")
    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"Classifier weights not found: {CLASSIFIER_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise SystemExit(f"Class names not found: {CLASS_NAMES_PATH}")

    print(f"Loading detector (DINOv2-base, output_attentions=True)...")
    detector = load_detector(args.device)

    print(f"Loading classifier (DINOv2-base + LoRA + linear)...")
    backbone, classifier, processor, class_names = load_classifier(args.device)
    print(f"  {len(class_names)} classes")

    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    print(f"Running pipeline (method={args.method})...")
    detections, viz_map = detect_and_classify(
        image, detector, backbone, classifier, processor, class_names,
        method=args.method, threshold=args.threshold, tau=args.tau,
        max_objects=args.max_objects, max_area_fraction=args.max_area_fraction,
        layer=args.layer, device=args.device,
    )

    print(f"Found {len(detections)} detections:")
    for i, d in enumerate(detections):
        x, y, w, h = d.box
        print(f"  {i}: ({x},{y}) {w}x{h}  -> {d.ingredient} ({d.confidence:.1%})")

    out_path = args.out / f"{args.image.stem}_{args.method}_pipeline.png"
    visualize_pipeline(image, detections, viz_map, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
