import os
import sys
from pathlib import Path

import numpy as np
from data_io import description_order, load_history
from style import (
    PRIMARY_DESC,
    add_series,
    new_single_axes,
    place_line_labels,
    save_figure,
    series_palette,
    style_axes,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
PLOT_DIR = REPO_ROOT / "results" / "plots"

# T * C_in
X_COL = "size"  

# columns plotted by each subplot - change here to re-route the Y axis
PERF_Y_COL    = "perf"     
RUNTIME_Y_COL = "gcycles"  
OI_Y_COL      = "oi_dram" 

MANUAL_LABEL_ADJUSTMENTS = {}


def _draw_lines(ax, df, palette, y_col, dropna=True):
    lines, all_y = [], []
    for desc in description_order(df):
        g = df[df["description"] == desc].sort_values(X_COL)
        if dropna:
            g = g.dropna(subset=[X_COL, y_col])
        g = g[np.isfinite(g[y_col])]
        if g.empty:
            continue
        is_primary = desc == PRIMARY_DESC
        lines.append(add_series(
            ax,
            x=g[X_COL], y=g[y_col],
            label=desc,
            primary=is_primary,
            color=palette[desc],
            linestyle="--" if is_primary else "-",
            linewidth=3.2 if is_primary else 2,
            marker="o" if is_primary else "s",
        ))
        all_y.extend(float(v) for v in g[y_col])
    return lines, all_y


def plot_performance_inputsize(df, palette):
    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = _draw_lines(ax, df, palette, PERF_Y_COL)
    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Performance: {app_config.MACHINE}",
        y_unit_text="Performance [flops / cycle]",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    save_figure(fig, PLOT_DIR / "performance_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'performance_inputsize'}")


def plot_runtime_inputsize(df, palette):
    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = _draw_lines(ax, df, palette, RUNTIME_Y_COL)
    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Runtime: {app_config.MACHINE}",
        y_unit_text="Time [Gcycles]",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    save_figure(fig, PLOT_DIR / "runtime_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'runtime_inputsize'}")


def plot_speedup_inputsize(df, palette):
    ref = df[df["description"] == PRIMARY_DESC]
    if ref.empty:
        print(f"No {PRIMARY_DESC} run in history - skipping speedup plot.")
        return
    ref_perf = ref.set_index(X_COL)[PERF_Y_COL]

    fig, ax = new_single_axes(figsize=(10, 6))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    lines, all_y = [], [1.0]

    for desc in description_order(df):
        if desc == PRIMARY_DESC:
            continue
        g = df[df["description"] == desc].sort_values(X_COL)
        g = g.dropna(subset=[X_COL, PERF_Y_COL])
        if g.empty:
            continue
        sizes = g[X_COL].values
        common = [s for s in sizes if s in ref_perf.index]
        if not common:
            continue
        speedup = g.set_index(X_COL).loc[common, PERF_Y_COL] / ref_perf.loc[common]
        lines.append(add_series(
            ax,
            x=common, y=speedup.values,
            label=desc,
            color=palette[desc],
            linestyle=":" if desc == "baseline" else "-",
            linewidth=2, marker="s",
        ))
        all_y.extend(float(v) for v in speedup)

    if not lines:
        return

    style_axes(
        ax,
        title=f"Speedup vs {PRIMARY_DESC}: {app_config.MACHINE}",
        y_unit_text="Speedup [x]",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.5, max(all_y) * 1.2)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    save_figure(fig, PLOT_DIR / "speedup_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'speedup_inputsize'}")


def plot_oi_inputsize(df, palette):
    fig, ax = new_single_axes(figsize=(10, 6))
    ridge = app_config.ROOFLINE_PI_VECTOR / app_config.ROOFLINE_BETA
    ax.axhline(ridge, color="gray", linestyle="--", linewidth=1.2, zorder=1,
               label=f"ridge = {ridge:.2f}")

    lines, all_y = _draw_lines(ax, df, palette, OI_Y_COL)
    if not lines:
        return

    style_axes(
        ax,
        title=f"Operational Intensity (L3): {app_config.MACHINE}",
        y_unit_text="Flops / byte",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, max(all_y) * 1.3)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    save_figure(fig, PLOT_DIR / "oi_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'oi_inputsize'}")


def generate_performance_plot():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl - skipping performance plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    palette = series_palette(description_order(df))

    plot_performance_inputsize(df, palette)
    plot_runtime_inputsize(df, palette)
    plot_speedup_inputsize(df, palette)
    plot_oi_inputsize(df, palette)
