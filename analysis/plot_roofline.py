import os
import sys
from pathlib import Path

import numpy as np
from data_io import description_order, load_history
from matplotlib.ticker import LogLocator
from style import (
    PALETTE,
    PRIMARY_DESC,
    add_series,
    new_single_axes,
    place_line_labels,
    save_figure,
    series_palette,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
PLOT_DIR = REPO_ROOT / "results" / "plots"

# X: operational intensity (flops / byte).  Y: perf (flops / cycle).
Y_COL = "perf"
ROOFLINES = [
    ("oi_dram", "roofline_dram", "Roofline (DRAM)"),
    ("oi_l3", "roofline_l3", "Roofline (L3 warm)"),
    ("oi_l2", "roofline_l2", "Roofline (L2 warm)"),
    ("oi_l1", "roofline_l1", "Roofline (L1 warm)"),
]


def _log_bounds(values, pad_powers=0.5, floor=1e-3):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return floor, 1.0
    span = 2**pad_powers
    lo = max(arr.min() / span, floor)
    hi = max(arr.max() * span, lo * 1.01)
    return lo, hi


def _draw_roofline(ax, xlim, ylim, beta, title):
    x_min, x_max = xlim
    y_min, y_max = ylim
    roofs = [
        ("π_s", app_config.ROOFLINE_PI_SCALAR, PALETTE["black"]),
        ("π_v", app_config.ROOFLINE_PI_VECTOR, PALETTE["black"]),
    ]

    ax.plot(
        [x_min, x_max],
        [beta * x_min, beta * x_max],
        color="black",
        linestyle="-",
        linewidth=1.5,
    )

    for label, peak, color in roofs:
        ridge = peak / beta
        ax.plot([ridge, x_max], [peak, peak], color=color, linestyle="-", linewidth=2.2)
        ax.plot([ridge, ridge], [y_min, peak], color="gray", linestyle=":", linewidth=1)
        ax.text(
            x_max * 0.97,
            peak * 1.1,
            f"{label} = {peak}",
            fontsize=9,
            color="black",
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    x_anchor = min(x_min * 2.0, x_max / 2.0)
    ax.annotate(
        f"β = {beta}",
        xy=(x_anchor, beta * x_anchor),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=9,
        color="black",
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(
        0.0,
        1.0,
        "Performance [flops / cycle]",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gray_text"],
        ha="left",
        va="bottom",
    )
    ax.set_ylabel("")
    ax.set_xlabel("Operational Intensity [flops / byte]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=1000))
    ax.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=1000))
    ax.grid(True, which="major", linewidth=0.6, color="white", alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _generate_one(df, palette, oi_col, output_name, title_base, beta):
    fig, ax = new_single_axes(figsize=(10, 6))
    lines = []

    for desc in description_order(df):
        g = df[df["description"] == desc].dropna(subset=[oi_col, Y_COL])
        g = g[
            (g[oi_col] > 0)
            & (g[Y_COL] > 0)
            & np.isfinite(g[oi_col])
            & np.isfinite(g[Y_COL])
        ]
        if g.empty:
            continue
        is_primary = desc == PRIMARY_DESC
        lines.append(
            add_series(
                ax,
                x=g[oi_col],
                y=g[Y_COL],
                label=desc,
                primary=is_primary,
                color=palette[desc],
                linestyle="--" if is_primary else "-",
                linewidth=3.0 if is_primary else 2,
                marker="o" if is_primary else "s",
            )
        )

    if not lines:
        return

    peaks = [app_config.ROOFLINE_PI_SCALAR, app_config.ROOFLINE_PI_VECTOR]
    ridges = [p / beta for p in peaks]

    all_x = list(ridges)
    all_y = list(peaks)
    for line in lines:
        all_x.extend(line.get_xdata(orig=False))
        all_y.extend(line.get_ydata(orig=False))

    xlim = _log_bounds(all_x, pad_powers=0.6)
    ylim = _log_bounds(all_y, pad_powers=0.4)

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = np.log2(x_max / x_min)
    y_span = np.log2(y_max / y_min)

    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    target_ratio = (bbox.height * fig_h) / (bbox.width * fig_w)
    current_ratio = y_span / x_span

    if current_ratio < target_ratio:
        y_mid = np.sqrt(y_min * y_max)
        half = x_span * target_ratio / 2
        y_min, y_max = y_mid / (2**half), y_mid * (2**half)
    elif current_ratio > target_ratio:
        x_mid = np.sqrt(x_min * x_max)
        half = y_span / target_ratio / 2
        x_min, x_max = x_mid / (2**half), x_mid * (2**half)

    _draw_roofline(
        ax, (x_min, x_max), (y_min, y_max), beta, f"{title_base}: {app_config.MACHINE}"
    )
    place_line_labels(lines)
    save_figure(fig, PLOT_DIR / output_name)
    print(f"Plot saved: {PLOT_DIR / output_name}")


def generate_roofline_plot():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl - skipping roofline plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    palette = series_palette(description_order(df))
    for oi_col, name, title in ROOFLINES:
        if oi_col not in df.columns:
            continue
        level = oi_col.replace("oi_", "")
        beta = app_config.roofline_beta_for(level)
        _generate_one(df, palette, oi_col, name, title, beta)
