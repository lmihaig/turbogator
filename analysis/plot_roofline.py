import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from data_io import load_history_dataframe, load_reference_dataframe
from matplotlib.ticker import LogLocator
from style import PALETTE, add_series, new_single_axes, place_line_labels, save_figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
REFERENCE_FILE = REPO_ROOT / "results" / "reference" / "metrics.json"
PLOT_DIR = REPO_ROOT / "results" / "plots"


def _roofline_points(df):
    n = pd.to_numeric(df["N"], errors="coerce")
    cycles = pd.to_numeric(df["cycles"], errors="coerce")
    flops = n.apply(app_config.calculate_total_flops)
    bytes_total = n.apply(app_config.calculate_total_bytes)

    points = pd.DataFrame({"I": flops / bytes_total, "perf": flops / cycles})
    valid = np.isfinite(points["I"]) & np.isfinite(points["perf"])
    valid &= (points["I"] > 0) & (points["perf"] > 0)
    return points[valid]


def _log_bounds(values, pad_powers=0.5, floor=1e-3):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return floor, 1.0

    span = 2**pad_powers
    lo = max(arr.min() / span, floor)
    hi = max(arr.max() * span, lo * 1.01)
    return lo, hi


def _draw_roofline(ax, xlim, ylim):
    beta = app_config.ROOFLINE_BETA
    roofs = [
        ("π_s", app_config.ROOFLINE_PI_SCALAR, PALETTE["green"]),
        ("π_v", app_config.ROOFLINE_PI_VECTOR, PALETTE["blue"]),
    ]

    x_min, x_max = xlim
    y_min, y_max = ylim

    x_mem = np.array([x_min, x_max], dtype=float)
    ax.plot(x_mem, beta * x_mem, color="black", linestyle="-", linewidth=1.5)

    for label, peak, color in roofs:
        ridge = peak / beta
        ax.plot([ridge, x_max], [peak, peak], color=color, linestyle="-", linewidth=2.2)
        ax.plot([ridge, ridge], [y_min, peak], color="gray", linestyle=":", linewidth=1)
        ax.text(
            x_max * 0.97,
            peak * 1.1,
            f"{label} = {peak}",
            fontsize=9,
            color=color,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    x_anchor = min(x_min * 2.0, x_max / 2.0)
    y_anchor = beta * x_anchor
    ax.annotate(
        f"β = {beta}",
        xy=(x_anchor, y_anchor),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=9,
        color="black",
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    ax.set_title(
        f"Roofline Model: {app_config.MACHINE}",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=16,
    )
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


def generate_roofline_plot():
    history_df = load_history_dataframe(HISTORY_FILE)
    reference_df = load_reference_dataframe(REFERENCE_FILE)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = new_single_axes(figsize=(10, 6))

    plot_lines = []

    reference_points = _roofline_points(reference_df)
    plot_lines.append(
        add_series(
            ax,
            x=reference_points["I"],
            y=reference_points["perf"],
            label="ezgatr",
            primary=True,
            linestyle="--",
            linewidth=3.0,
            marker="o",
        )
    )

    for _, run in history_df.iterrows():
        desc = str(run.get("description", "run")).strip()
        points = run.get("data", [])

        run_points = _roofline_points(pd.DataFrame(points))

        plot_lines.append(
            add_series(
                ax,
                x=run_points["I"],
                y=run_points["perf"],
                label=desc,
                linewidth=2,
                marker="s",
            )
        )

    beta = app_config.ROOFLINE_BETA
    peaks = [app_config.ROOFLINE_PI_SCALAR, app_config.ROOFLINE_PI_VECTOR]
    ridges = [peak / beta for peak in peaks]

    all_x = list(ridges)
    all_y = list(peaks)
    for line in plot_lines:
        all_x.extend(line.get_xdata(orig=False))
        all_y.extend(line.get_ydata(orig=False))

    xlim = _log_bounds(all_x, pad_powers=0.6)
    ylim = _log_bounds(all_y, pad_powers=0.4)

    # Match log spans to axes
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = np.log2(x_max / x_min)
    y_span = np.log2(y_max / y_min)

    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    axes_w = bbox.width * fig_w
    axes_h = bbox.height * fig_h
    target_ratio = axes_h / axes_w
    current_ratio = y_span / x_span

    # Expand ranges to match axes aspect
    if current_ratio < target_ratio:
        y_mid = np.sqrt(y_min * y_max)
        half = x_span * target_ratio / 2
        y_min = y_mid / (2**half)
        y_max = y_mid * (2**half)
    elif current_ratio > target_ratio:
        x_mid = np.sqrt(x_min * x_max)
        half = y_span / target_ratio / 2
        x_min = x_mid / (2**half)
        x_max = x_mid * (2**half)

    xlim = (x_min, x_max)
    ylim = (y_min, y_max)

    _draw_roofline(ax, xlim, ylim)

    place_line_labels(plot_lines)

    output_path = PLOT_DIR / "roofline.pdf"
    save_figure(fig, output_path)
    print(f"Roofline plot successfully generated at: {output_path}")


if __name__ == "__main__":
    generate_roofline_plot()
