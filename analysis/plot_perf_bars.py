import math
import os
import sys
from pathlib import Path

import numpy as np
from data_io import description_order, load_history
from style import annotate_bar_values, new_figure, save_figure, series_palette

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
PLOT_DIR = REPO_ROOT / "results" / "plots"

# (column in dataframe, display label, subtitle)
METRICS = [
    ("ipc", "IPC", "instructions / cycle  ↑ better"),
    ("avx2_ratio", "AVX2 ratio", "AVX2 FLOPs / total FLOPs  ↑ more vectorized"),
    ("l1_load_miss_rate", "L1 load miss rate", "L1↔L2 lines / L1 loads"),
    ("l2_miss_rate", "L2 miss rate", "L2→L3 lines / L1→L2 lines"),
    ("l3_demand_miss_rate", "L3 demand miss", "L3 demand misses / L1 loads"),
    ("store_load_ratio", "Store/Load", "L1 stores / L1 loads"),
    ("dram_per_gflop", "DRAM bytes/GFLOP", "DRAM bytes / GFLOP  ↓ more compute-bound"),
    ("dram_prefetch_share", "DRAM prefetch %", "1 − demand-L3-miss bytes / DRAM bytes"),
    ("bytes_per_flop_l1", "Bytes/FLOP L1", "L1↔L2 bytes / FLOP  ↑ memory-bound at L1"),
]


def generate_perf_bars():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl — skipping perf bars.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    target_n = app_config.REPRESENTATIVE_N
    palette = series_palette(description_order(df))

    # one row per description, picking the run at the target N (closest if exact missing)
    rows = []
    for desc in description_order(df):
        g = df[df["description"] == desc]
        if g.empty:
            continue
        exact = g[g["N"] == target_n]
        row = (
            exact.iloc[0]
            if not exact.empty
            else g.iloc[(g["N"] - target_n).abs().argsort().iloc[0]]
        )
        rows.append((desc, row))

    # drop metrics whose column is missing or has no finite values
    metrics = [
        (c, lbl, sub)
        for c, lbl, sub in METRICS
        if c in df.columns and any(np.isfinite(r[c]) for _, r in rows)
    ]
    if not rows or not metrics:
        print("No hardware perf data found — skipping perf bars.")
        return

    n_metrics, n_descs = len(metrics), len(rows)

    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = new_figure(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    bar_width = 0.7
    for ax, (col, label, subtitle) in zip(axes, metrics):
        for i, (desc, row) in enumerate(rows):
            val = float(row[col]) if np.isfinite(row[col]) else float("nan")
            bars = ax.bar(
                i, val, width=bar_width, color=palette[desc], label=desc, zorder=3
            )
            annotate_bar_values(ax, bars, decimals=3)

        ax.set_xticks(np.arange(n_descs))
        ax.set_xticklabels([d for d, _ in rows], rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=7, color="0.5")
        ax.grid(axis="y", linewidth=0.6, color="white", zorder=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, n_descs - 0.5)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Hardware Profile at N={target_n} — {app_config.MACHINE}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    save_figure(fig, PLOT_DIR / "perf_bars", tight_rect=(0, 0.02, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'perf_bars'}")
