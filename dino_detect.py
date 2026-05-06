"""
DINOv2-based unsupervised object detection.

Two methods:
  - threshold: quantile-threshold the [CLS]->patch attention map, bounding
    boxes from connected components.
  - tokencut: spectral graph partitioning on patch features (Wang et al. 2022,
    "Self-Supervised Transformers for Unsupervised Object Discovery using
    Normalized Cut"). Iteratively bipartitions the patch graph to discover
    multiple foreground objects, implemented as an improvement over the
    attention thresholding approach 

Usage:
    python dino_detect.py fridge_data/fridge_test1.jpg
    python dino_detect.py fridge_data/fridge_test1.jpg --method tokencut
    python dino_detect.py <image> --method tokencut --tau 0.15 --max-objects 8
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
DEFAULT_TOKENCUT_TAU = 0.2
DEFAULT_TOKENCUT_MAX_OBJECTS = 6

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
    # Skip the standard processor (which center-crops to 224) so we keep the whole image.
    # 448 = 32x32 patch grid (size must be a multiple of patch_size=14).
    image_resized = image.resize((size, size), Image.BICUBIC)  
    arr = np.array(image_resized).astype(np.float32) / 255.0  
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD  
    return tensor.unsqueeze(0)  # add batch dim: (C,H,W) -> (1,C,H,W)


def get_patch_features(model, image_tensor: torch.Tensor) -> np.ndarray:
    """Returns (n_patches, hidden_dim) patch features from DINOv2's last layer.

    Used by TokenCut as the input to the patch-affinity graph.
    """
    with torch.no_grad():  
        outputs = model(pixel_values=image_tensor)
        
    return outputs.last_hidden_state[0, 1:].cpu().numpy()


def get_attention_map(model, image_tensor: torch.Tensor, layer: int = -1) -> np.ndarray:
    """Mean of [CLS] -> patch attention at `layer`, averaged over heads.

    layer = -1 (last) is most semantic but spatially crude,
    earlier layers often have finer spatial detail.
    """
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor, output_attentions=True)

    # outputs.attentions is a tuple of length n_layers, each entry has shape
    # (batch=1, n_heads=12, seq_len, seq_len). [layer][0] -> picks layer + drops batch dim,
    # giving (n_heads, seq_len, seq_len).
    attn = outputs.attentions[layer][0]
    # Each value is "how much the CLS token attends to patch j".
    cls_to_patch = attn[:, 0, 1:]
    cls_to_patch = cls_to_patch.mean(dim=0)

    # Patches form a square grid (32x32 for our 448x448 input). Reshape (n_patches,) -> (32, 32)
    # so we can upsample it back to image resolution and threshold spatially.
    n_patches = cls_to_patch.shape[0]
    grid = int(np.sqrt(n_patches))
    assert grid * grid == n_patches, f"Non-square patch grid: {n_patches}"
    return cls_to_patch.reshape(grid, grid).cpu().numpy()


def attention_to_boxes(attn_map, image_size, quantile_threshold,
                        min_area_fraction, max_area_fraction):
    """Threshold attention -> connected components -> bounding boxes."""
    H, W = image_size
    # Upsample the small (32x32) attention map to full image resolution so boxes
    # can be drawn in original-image coordinates. 
    attn_up = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Quantile threshold: e.g., 0.6 means "anything above the 60th percentile is foreground".
    threshold_value = float(np.quantile(attn_up, quantile_threshold))
    binary = (attn_up > threshold_value).astype(np.uint8)  

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = H * W
    boxes = []
    for i in range(1, n_labels):  # skip background (i=0)
        x, y, w, h, area = stats[i]
        # Filter by both component pixel area (to drop noise) and bounding-box area
        # (drop "super-boxes" that span/encompass most of the image).
        box_area = w * h
        if area < image_area * min_area_fraction:  # too small 
            continue
        if box_area > image_area * max_area_fraction:  # too large 
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes, attn_up


def tokencut_to_boxes(
    features: np.ndarray,
    image_size: tuple[int, int],
    grid_size: int,
    tau: float,
    min_area_fraction: float,
    max_area_fraction: float,
    max_objects: int,
):
    """Multi-object TokenCut.

    Iteratively bipartitions a thresholded patch-affinity graph using the
    second-smallest eigenvector of the normalized Laplacian. Each iteration
    discovers one foreground region. That region is then removed and the next
    iteration runs on the remaining patches.

    Returns (boxes, fg_grid) where:
      - boxes: list of (x, y, w, h) in original image coordinates
      - fg_grid: (grid_size, grid_size) float mask, 1.0 = foreground
    """
    H, W = image_size
    n_patches = features.shape[0]  
    image_area = H * W

    # Build patch-affinity graph: edge between two patches if their feature cosine similarity > tau.
    # Binary thresholding follows the original TokenCut paper (Wang et al. 2022).
    
    # L2-normalize each feature vector so dot product == cosine similarity.
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8  
    features_n = features / norm
    # pairwise dot product = cosine similarity matrix, shape (n_patches, n_patches).
    affinity = features_n @ features_n.T
    # threshold to binary (1 = connected, 0 = not)
    affinity = (affinity > tau).astype(np.float32)
    # Zero diagonal, a patch shouldn't have a self-loop in the graph.
    np.fill_diagonal(affinity, 0)


    available = np.ones(n_patches, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    full_fg = np.zeros(n_patches, dtype=np.uint8)  # tracks union of all discovered foregrounds

    for _ in range(max_objects): 
        active = np.where(available)[0]
        if len(active) < 8:  # too few patches left to do meaningful spectral partitioning
            break

        # Build normalized graph Laplacian on the currently-active patches.
        sub_aff = affinity[np.ix_(active, active)]
        d = sub_aff.sum(axis=1) + 1e-5
        D = np.diag(d)            # diagonal degree matrix
        L = D - sub_aff           # unnormalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        # Normalized Laplacian: D^(-1/2) (D - W) D^(-1/2)
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        try:
            # eigh returns eigenvalues (ascending) and eigenvectors of a symmetric matrix.
            _, evecs = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            break  
        if evecs.shape[1] < 2:
            break
    
        second = evecs[:, 1]

        side_a = second > 0
        side_b = ~side_a
        # Foreground = smaller side. Objects are usually a smaller portion than background.
        fg_local = side_a if 0 < side_a.sum() <= side_b.sum() else side_b
        if fg_local.sum() < 2:
            break  # tiny "object" equals noise, stop

        # Lift fg_local  back into the full n_patches space.
        fg_global = np.zeros(n_patches, dtype=bool)
        fg_global[active[fg_local]] = True
        # Reshape flat (n_patches,) -> 2D grid, then upsample to image resolution.
        fg_2d = fg_global.reshape(grid_size, grid_size).astype(np.uint8)
        fg_up = cv2.resize(fg_2d, (W, H), interpolation=cv2.INTER_NEAREST)

        # Same connected-components, bounding box flow as in the threshold method.
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg_up, connectivity=8)
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            box_area = w * h
            if area < image_area * min_area_fraction:
                continue
            if box_area > image_area * max_area_fraction:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))

        # Mask out the foreground patches before the next iteration so we discover
        # different objects each pass.
        full_fg |= fg_global.astype(np.uint8)
        available[fg_global] = False
        if available.sum() < 0.1 * n_patches:
            break

    fg_grid = full_fg.reshape(grid_size, grid_size).astype(np.float32)
    return boxes, fg_grid


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
    parser = argparse.ArgumentParser(description="DINOv2-based unsupervised object detection")
    parser.add_argument("image", type=Path)
    parser.add_argument("--method", default="threshold", choices=["threshold", "tokencut"],
                        help="Detection method")
    # threshold-method params
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="[threshold] Attention quantile (0-1, higher = stricter)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="[threshold] Attention layer index (-1=last, 0-11 for DINOv2-base)")
    # tokencut-method params
    parser.add_argument("--tau", type=float, default=DEFAULT_TOKENCUT_TAU,
                        help="[tokencut] Affinity threshold (cosine similarity)")
    parser.add_argument("--max-objects", type=int, default=DEFAULT_TOKENCUT_MAX_OBJECTS,
                        help="[tokencut] Max iterations (max objects discovered)")
    # shared params
    parser.add_argument("--max-area-fraction", type=float, default=MAX_BOX_AREA_FRACTION,
                        help="Drop boxes covering more than this fraction of the image")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    print(f"Loading DINOv2-base on {args.device} (method={args.method})...")
    model = load_model(args.device)

    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")
    image_tensor = preprocess(image).to(args.device)
    H, W = image.size[1], image.size[0]

    if args.method == "threshold":
        attn_map = get_attention_map(model, image_tensor, layer=args.layer)
        print(f"Attention map: {attn_map.shape} (layer {args.layer})")
        boxes, viz_map = attention_to_boxes(
            attn_map, (H, W),
            quantile_threshold=args.threshold,
            min_area_fraction=MIN_BOX_AREA_FRACTION,
            max_area_fraction=args.max_area_fraction,
        )
    else:  # tokencut is used 
        features = get_patch_features(model, image_tensor)
        n_patches = features.shape[0]
        grid_size = int(np.sqrt(n_patches))
        assert grid_size * grid_size == n_patches, f"Non-square patch grid: {n_patches}"
        print(f"Patch features: {features.shape} (grid {grid_size}x{grid_size})")
        boxes, fg_grid = tokencut_to_boxes(
            features, (H, W),
            grid_size=grid_size,
            tau=args.tau,
            min_area_fraction=MIN_BOX_AREA_FRACTION,
            max_area_fraction=args.max_area_fraction,
            max_objects=args.max_objects,
        )
        viz_map = cv2.resize(fg_grid, (W, H), interpolation=cv2.INTER_NEAREST)

    print(f"Found {len(boxes)} bounding boxes")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  box {i}: ({x},{y}) {w}x{h}")

    out_path = args.out / f"{args.image.stem}_{args.method}_detected.png"
    visualize(image, boxes, viz_map, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
