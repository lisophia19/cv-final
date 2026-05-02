"""
Compare DINO (LoRA head) predictions to YOLO labels on every crop under runs/eval/crops.

Requires crop_manifest.json alongside the images (written by yolo.test_model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CROPS_DIR = REPO_ROOT / "runs" / "eval" / "crops"


def _first_existing(*candidates: Path, kind: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {kind}; tried: " + ", ".join(str(p) for p in candidates))


def load_dino_classifier(device: torch.device):
    data_yaml = REPO_ROOT / "ingredients_data" / "data.yaml"
    with open(data_yaml, "r", encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["names"]

    adapter_dir = _first_existing(
        REPO_ROOT / "dinov2_lora_adapters",
        REPO_ROOT / "dino" / "lora" / "dinov2_lora_adapters",
        kind="LoRA adapter folder",
    )
    clf_path = _first_existing(
        REPO_ROOT / "dinov2_lora_classifier.pt",
        REPO_ROOT / "dino" / "lora" / "dinov2_lora_classifier.pt",
        kind="dinov2_lora_classifier.pt",
    )

    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone = PeftModel.from_pretrained(backbone, str(adapter_dir))
    embed_dim = backbone.config.hidden_size
    classifier = nn.Linear(embed_dim, len(class_names)).to(device)
    classifier.load_state_dict(torch.load(clf_path, map_location=device))
    backbone.eval()
    classifier.eval()
    return processor, backbone, classifier, class_names


def dino_predict_crop(
    pil_rgb: Image.Image,
    processor,
    backbone,
    classifier,
    device: torch.device,
    class_names: list,
) -> tuple[str, float]:
    pixel_values = processor(images=pil_rgb, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        emb = backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        logits = classifier(emb)
        probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = probs.argmax().item()
    return class_names[pred_idx], float(probs[pred_idx].item())


def compare_yolo_dino_on_crops(
    crops_dir: Path | None = None,
    *,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    crops_dir = Path(crops_dir or DEFAULT_CROPS_DIR).resolve()
    manifest_path = crops_dir / "crop_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run yolo.test_model() on your images to regenerate crops and manifest."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest: dict[str, dict] = json.load(f)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, backbone, classifier, class_names = load_dino_classifier(device)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in crops_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    rows: list[tuple[str, str, str, float, bool]] = []
    missing = 0

    for fp in files:
        key = fp.name
        if key not in manifest:
            missing += 1
            continue

        yolo_label = manifest[key]["yolo_label"]
        pil = Image.open(fp).convert("RGB")
        dino_label, dino_conf = dino_predict_crop(
            pil, processor, backbone, classifier, device, class_names
        )
        agree = yolo_label == dino_label
        rows.append((key, yolo_label, dino_label, dino_conf, agree))

    matched = sum(1 for r in rows if r[4])
    total = len(rows)
    agreement = matched / total if total else 0.0

    for key, yl, dl, dc, agree in rows:
        if verbose or not agree:
            mark = "OK" if agree else "DIFF"
            print(f"[{mark}] {key}: YOLO={yl!r} DINO={dl!r} ({dc:.1%})")

    print(f"\nCrops with manifest entries: {total}")
    print(f"YOLO vs DINO agreement: {matched}/{total} = {agreement:.1%}")
    if missing:
        print(f"Files skipped (no manifest entry): {missing}")

    return {
        "total": total,
        "matched": matched,
        "agreement": agreement,
        "missing_manifest": missing,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO vs DINO label agreement on eval crops.")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help=f"Folder with crop images + crop_manifest.json (default: {DEFAULT_CROPS_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every crop; default is mismatches only plus summary.",
    )
    args = parser.parse_args()
    compare_yolo_dino_on_crops(crops_dir=args.crops_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
