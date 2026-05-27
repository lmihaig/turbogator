import os
import sys
from pathlib import Path

import numpy as np
from data_io import description_order, load_history
from style import PALETTE, new_figure, save_figure, series_palette

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
PLOT_DIR = REPO_ROOT / "results" / "plots"

STACKS = [
    ("fp_scalar", 1, "scalar (x1)", PALETTE["blue"]),
    ("fp_128b", 4, "128b packed (x4)", PALETTE["green"]),
    ("fp_256b", 8, "256b packed (x8)", PALETTE["orange"]),
]


def generate_flop_decomposition():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl - skipping flop decomposition plot.")
        return

    descs = description_order(df)
    available = [(c, w, lbl, col) for c, w, lbl, col in STACKS if c in df.columns]
    if not descs or not available:
        print("No FP_ARITH columns in history - skipping flop decomposition.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    desc_palette = series_palette(descs)

    cols = min(len(descs), 3)
    rows = (len(descs) + cols - 1) // cols
    fig, axes = new_figure(
        nrows=rows, ncols=cols, figsize=(4.2 * cols, 3.6 * rows), sharey=True
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, desc in zip(axes_flat, descs):
        sub = df[df["description"] == desc].sort_values("N")
        if sub.empty:
            ax.set_visible(False)
            continue

        ns = sub["N"].astype(int).tolist()
        x = np.arange(len(ns))
        weighted = {c: w * sub[c].fillna(0).values for c, w, _, _ in available}
        total = sum(weighted.values())

        bottom = np.zeros_like(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for c, _, lbl, color in available:
                pct = np.where(total > 0, 100.0 * weighted[c] / total, 0.0)
                ax.bar(
                    x,
                    pct,
                    bottom=bottom,
                    color=color,
                    label=lbl,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                bottom = bottom + pct

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in ns], fontsize=8)
        ax.set_title(desc, fontsize=11, fontweight="bold", color=desc_palette[desc])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linewidth=0.6, color="white", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ylabel only on left column
    for r in range(rows):
        if r * cols < len(descs):
            axes_flat[r * cols].set_ylabel("% of flops_pmc")

    # hide unused panels
    for ax in axes_flat[len(descs) :]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(available),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"FLOP source breakdown - {app_config.MACHINE}",
        fontsize=12,
        fontweight="bold",
        y=1.07,
    )

    save_figure(fig, PLOT_DIR / "flop_decomposition", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'flop_decomposition'}")
