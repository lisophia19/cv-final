"""
DINOv2 attention-based bounding box detector.

Uses self-supervised attention maps from DINOv2 to discover object regions
without supervision then converts the attention map to bounding boxes via
quantile thresholding and connected-component labeling.

Usage:
    python dino_detect.py fridge_data/fridge_test1.jpg
    python dino_detect.py <image> --threshold 0.7 --out runs/dino_detect/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "facebook/dinov2-base"
DEFAULT_OUT_DIR = BASE_DIR / "runs" / "dino_detect"
PATCH_SIZE = 14
INPUT_SIZE = 448 
DEFAULT_THRESHOLD = 0.6
MIN_BOX_AREA_FRACTION = 0.005
MAX_BOX_AREA_FRACTION = 0.7

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model(device: str = "cpu"):
    model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess(image: Image.Image, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resizes to size x size and ImageNet-normalize. Returns (1, 3, size, size)."""
    image_resized = image.resize((size, size), Image.BICUBIC)
    arr = np.array(image_resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def get_attention_map(model, image_tensor: torch.Tensor, layer: int = -1) -> np.ndarray:
    """Mean of [CLS] -> patch attention at `layer`, averaged over heads.

    layer = -1 (last) is most semantic but spatially coarse;
    earlier layers (e.g. 6-8) often have finer spatial detail.
    """
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor, output_attentions=True)

    attn = outputs.attentions[layer][0]  
    cls_to_patch = attn[:, 0, 1:]     
    cls_to_patch = cls_to_patch.mean(dim=0)  

    n_patches = cls_to_patch.shape[0]
    grid = int(np.sqrt(n_patches))
    assert grid * grid == n_patches, f"Non-square patch grid: {n_patches}"
    return cls_to_patch.reshape(grid, grid).cpu().numpy()


def attention_to_boxes(attn_map, image_size, quantile_threshold,
                        min_area_fraction, max_area_fraction):
    """Threshold attention -> connected components -> bounding boxes."""
    H, W = image_size
    attn_up = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    threshold_value = float(np.quantile(attn_up, quantile_threshold))
    binary = (attn_up > threshold_value).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = H * W
    boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        box_area = w * h
        if area < image_area * min_area_fraction:
            continue
        if box_area > image_area * max_area_fraction:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes, attn_up


def visualize(image, boxes, attn_map, out_path):
    """Saves a 3-panel visualization: original | heatmap overlay | drawn boxes."""
    img_np = np.array(image)

    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap_rgb).astype(np.uint8)

    boxed = img_np.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)

    combined = np.concatenate([img_np, overlay, boxed], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="DINOv2 attention-based object detection")
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Attention quantile threshold (0-1, higher = stricter)")
    parser.add_argument("--max-area-fraction", type=float, default=MAX_BOX_AREA_FRACTION,
                        help="Drop boxes covering more than this fraction of the image")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Attention layer index (-1=last, 0-11 for DINOv2-base)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    print(f"Loading DINOv2-base on {args.device}...")
    model = load_model(args.device)

    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    image_tensor = preprocess(image).to(args.device)
    attn_map = get_attention_map(model, image_tensor, layer=args.layer)
    print(f"Attention map: {attn_map.shape} (layer {args.layer})")

    H, W = image.size[1], image.size[0]
    boxes, attn_up = attention_to_boxes(
        attn_map, (H, W),
        quantile_threshold=args.threshold,
        min_area_fraction=MIN_BOX_AREA_FRACTION,
        max_area_fraction=args.max_area_fraction,
    )

    print(f"Found {len(boxes)} bounding boxes")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  box {i}: ({x},{y}) {w}x{h}")

    out_path = args.out / f"{args.image.stem}_detected.png"
    visualize(image, boxes, attn_up, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
