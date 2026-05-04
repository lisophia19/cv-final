#!/usr/bin/env python3
"""
Plot mean detection precision / recall / F1 from an eval_e2e*.json summary.

Uses the same JSON as the top-k poster figure so all numbers stay consistent
with one evaluation run (no recompute).

Usage:
    python3 scripts/plot_poster_detection_prf1.py --json runs/e2e_eval/e2e_eval_20260429T152250Z.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True, help="E2E eval JSON (eval_e2e.py output).")
    p.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "figures" / "poster_detection_prf1.png",
    )
    p.add_argument("--title", default="Detection vs labels (mean over eval set)")
    args = p.parse_args()

    path = args.json.resolve()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    agg = data["aggregate"]
    pr = float(agg["mean_precision"])
    rc = float(agg["mean_recall"])
    f1 = float(agg["mean_f1"])
    n = int(agg["n_images"])

    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=150)
    labels = ("Precision", "Recall", "F1")
    vals = (pr, rc, f1)
    colors = ("#2c5282", "#2f855a", "#744210")
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(v + 0.03, 0.98),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{args.title} (n={n})")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out} (from {path.name})")


if __name__ == "__main__":
    main()
