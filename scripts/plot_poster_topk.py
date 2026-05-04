#!/usr/bin/env python3
"""
Plot retrieval top-k hit rates from an eval_e2e.py (or eval_e2e_dino.py) JSON summary.

For the team poster: bar chart of top-1 / top-3 / top-5 match rate vs. reasonable_recipes.

Usage:
    python scripts/plot_poster_topk.py
    python scripts/plot_poster_topk.py --json runs/e2e_eval/e2e_eval_20260429T152250Z.json
    python scripts/plot_poster_topk.py --json a.json --json b.json --labels baseline tuned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def load_rates(path: Path) -> tuple[float, float, float]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    agg = data["aggregate"]
    return float(agg["top1_rate"]), float(agg["top3_rate"]), float(agg["top5_rate"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top-k retrieval rates from E2E eval JSON.")
    parser.add_argument(
        "--json",
        dest="json_paths",
        action="append",
        default=None,
        help="E2E eval summary JSON (repeat for multiple series). Default: latest under runs/e2e_eval/.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels for each --json (same order).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "figures" / "poster_topk_retrieval.png",
        help="Output PNG path.",
    )
    parser.add_argument("--title", type=str, default="Recipe retrieval (reasonable_recipes)")
    args = parser.parse_args()

    if args.json_paths:
        paths = [Path(p).resolve() for p in args.json_paths]
    else:
        out_dir = BASE_DIR / "runs" / "e2e_eval"
        candidates = sorted(out_dir.glob("e2e_eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(f"No e2e_eval_*.json found under {out_dir}. Run eval_e2e.py first.")
        paths = [candidates[0]]

    series = [load_rates(p) for p in paths]
    labels = args.labels if args.labels else [p.stem for p in paths]
    if len(labels) != len(series):
        raise SystemExit("--labels count must match --json count.")

    x = (1, 3, 5)
    width = 0.22 if len(series) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    offsets = [-width * (len(series) - 1) / 2 + i * width for i in range(len(series))]

    for off, lab, (r1, r3, r5) in zip(offsets, labels, series):
        ys = (r1, r3, r5)
        bars = ax.bar([xi + off for xi in x], ys, width=width * 0.95, label=lab)
        for bar, y in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(y + 0.02, 0.98),
                f"{y:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Top-1", "Top-3", "Top-5"])
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(args.title)
    ax.axhline(0, color="#333", linewidth=0.8)
    if len(series) > 1:
        ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
