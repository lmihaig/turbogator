import os
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data_io import description_order, load_history
from style import (
    PRIMARY_DESC,
    add_series,
    new_figure,
    new_single_axes,
    place_dot_labels,
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
PERF_Y_COL = "perf"
PERF_THEORY_Y_COL = "perf_theory"
RUNTIME_Y_COL = "gcycles"
IPC_Y_COL = "ipc"


OI_LEVELS = [
    ("oi_l1", "L1"),
    ("oi_l2", "L2"),
    ("oi_l3", "L3"),
    ("oi_dram", "DRAM"),
]

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
        lines.append(
            add_series(
                ax,
                x=g[X_COL],
                y=g[y_col],
                label=desc,
                primary=is_primary,
                color=palette[desc],
                linestyle="--" if is_primary else "-",
                linewidth=3.2 if is_primary else 2,
                marker="o" if is_primary else "s",
            )
        )
        all_y.extend(float(v) for v in g[y_col])
    return lines, all_y


CACHE_LINE_COLOR = "0.35"


def add_cache_lines(ax):
    if not getattr(app_config, "SHOW_CACHE_LINES", False):
        return
    xmin, xmax = ax.get_xlim()
    trans = ax.get_xaxis_transform()

    for name, x in app_config.CACHE_LINE_SIZES.items():
        if not (xmin <= x <= xmax):
            continue
        ax.axvline(x, color=CACHE_LINE_COLOR, linestyle=":", linewidth=1.5, zorder=1)
        ax.text(
            x,
            0.94,
            f"{name} cache",
            transform=trans,
            rotation=0,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=CACHE_LINE_COLOR,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=CACHE_LINE_COLOR,
                lw=0.6,
                alpha=0.85,
            ),
        )


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
    add_cache_lines(ax)
    save_figure(fig, PLOT_DIR / "performance_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'performance_inputsize'}")


def plot_performance_inputsize_theory(df, palette):
    if PERF_THEORY_Y_COL not in df.columns:
        print(f"Column {PERF_THEORY_Y_COL!r} not in data - skipping theory perf plot.")
        return
    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = _draw_lines(ax, df, palette, PERF_THEORY_Y_COL)
    if not lines:
        return
    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Performance (analytic FLOPs): {app_config.MACHINE}",
        y_unit_text="Performance [theoretical flops / cycle]",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    add_cache_lines(ax)
    save_figure(
        fig, PLOT_DIR / "performance_inputsize_theory", tight_rect=(0, 0, 1.0, 1.0)
    )
    print(f"Plot saved: {PLOT_DIR / 'performance_inputsize_theory'}")


def plot_performance_inputsize_ipc(df, palette):
    if IPC_Y_COL not in df.columns:
        print(f"Column {IPC_Y_COL!r} not in data - skipping IPC plot.")
        return
    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = _draw_lines(ax, df, palette, IPC_Y_COL)
    if not lines:
        return
    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"IPC: {app_config.MACHINE}",
        y_unit_text="IPC [instructions / cycle]",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    add_cache_lines(ax)
    save_figure(
        fig, PLOT_DIR / "performance_inputsize_ipc", tight_rect=(0, 0, 1.0, 1.0)
    )
    print(f"Plot saved: {PLOT_DIR / 'performance_inputsize_ipc'}")


def plot_runtime_inputsize(df, palette):
    has_baseline = "baseline" in df["description"].values
    nrows = 2 if has_baseline else 1

    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(10, 6 * nrows), squeeze=False
    )

    plot_configs = [(df, " (With Baseline)" if has_baseline else "")]
    if has_baseline:
        df_no_baseline = df[df["description"] != "baseline"]
        plot_configs.append((df_no_baseline, " (Without Baseline)"))

    for ax, (data, title_suffix) in zip(axes.flatten(), plot_configs):
        lines, all_y = _draw_lines(ax, data, palette, RUNTIME_Y_COL)
        y_max = max(all_y) if all_y else 1.0

        style_axes(
            ax,
            title=f"Runtime: {app_config.MACHINE}{title_suffix}",
            y_unit_text="Time [Gcycles]",
            x_label="Input Size (TxC_in)",
            x_scale="log2",
            grid_axis="y",
            ylim=(0, max(1.0, y_max * 1.2)),
            hide_left_spine=True,
        )
        place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
        add_cache_lines(ax)

    save_figure(fig, PLOT_DIR / "runtime_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'runtime_inputsize'}")


def plot_speedup_inputsize(df, palette):
    ref = df[df["description"] == PRIMARY_DESC]
    if ref.empty:
        print(f"No {PRIMARY_DESC} run in history - skipping speedup plot.")
        return
    ref_perf = ref.set_index(X_COL)[PERF_Y_COL]

    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = [], []

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
        lines.append(
            add_series(
                ax,
                x=common,
                y=speedup.values,
                label=desc,
                color=palette[desc],
                linestyle=":" if desc == "baseline" else "-",
                linewidth=2,
                marker="s",
            )
        )
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
    add_cache_lines(ax)
    save_figure(fig, PLOT_DIR / "speedup_inputsize", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'speedup_inputsize'}")


def plot_oi_inputsize(df, palette, oi_col, level_label):
    if oi_col not in df.columns:
        print(f"Column {oi_col!r} not in data - skipping OI ({level_label}) plot.")
        return

    fig, ax = new_single_axes(figsize=(10, 6))
    lines, all_y = _draw_lines(ax, df, palette, oi_col)
    if not lines:
        return

    style_axes(
        ax,
        title=f"Operational Intensity ({level_label}): {app_config.MACHINE}",
        y_unit_text="Flops / byte",
        x_label="Input Size (TxC_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, max(all_y) * 1.3)),
        hide_left_spine=True,
    )
    place_line_labels(lines, label_adjustments=MANUAL_LABEL_ADJUSTMENTS)
    add_cache_lines(ax)
    name = f"oi_inputsize_{level_label.lower()}"
    save_figure(fig, PLOT_DIR / name, tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / name}")


def plot_work_efficiency(df, palette):
    need = {"flops_efficiency", "gcycles", "perf", "perf_theory"}
    if not need.issubset(df.columns):
        print("Missing work/efficiency columns - skipping work-efficiency plot.")
        return

    baseline_df = df[df["description"] == "baseline"]
    if baseline_df.empty:
        print("No 'baseline' found - skipping work-efficiency plot.")
        return

    n_vals = sorted(int(n) for n in baseline_df["N"].dropna().unique())
    if not n_vals:
        return

    n_low = n_vals[0]
    n_high = n_vals[-1]
    n_target = 8 if 8 in n_vals else min(n_vals, key=lambda n: abs(n - 8))

    target_idx = n_vals.index(n_target)
    high_idx = n_vals.index(n_high)
    if high_idx - target_idx > 1:
        n_mid = n_vals[target_idx + (high_idx - target_idx) // 2]
    else:
        n_mid = next((n for n in n_vals if n not in (n_low, n_target, n_high)), None)

    unique_targets = []
    for t in [n_low, n_target, n_mid, n_high]:
        if t is not None and t not in unique_targets:
            unique_targets.append(t)

    while len(unique_targets) < 4:
        unique_targets.append(None)

    fig, axes = new_figure(nrows=2, ncols=2, figsize=(12, 9), squeeze=False)

    for ax, target_n in zip(axes.ravel(), unique_targets):
        if target_n is None:
            ax.set_visible(False)
            continue

        pts = []
        for desc in description_order(df):
            g = df[df["description"] == desc]
            if g.empty:
                continue

            exact = g[g["N"] == target_n]
            if exact.empty:
                continue

            row = exact.iloc[0]
            eff, gc = row["flops_efficiency"], row["gcycles"]

            if np.isfinite(eff) and np.isfinite(gc) and gc > 0:
                pts.append((desc, row))

        if not pts:
            ax.set_visible(False)
            continue

        ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
        ax.text(
            0.99,
            101.0,
            "100%",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.45",
        )

        dot_pts = []
        for desc, row in pts:
            x = float(row["gcycles"])
            y = float(row["flops_efficiency"]) * 100.0
            is_primary = desc == PRIMARY_DESC
            ax.scatter(
                x,
                y,
                s=260 if is_primary else 150,
                color=palette[desc],
                edgecolor="black",
                linewidth=1.1,
                zorder=4,
                marker="o" if is_primary else "s",
            )
            dot_pts.append((desc, x, y, palette[desc]))

        ax.set_xscale("log", base=2)
        style_axes(
            ax,
            title=f"Work vs. Speed at N={target_n}",
            y_unit_text="Algorithmic FLOPs executed  [% of theory]",
            x_label="Runtime [Gcycles]",
            x_scale="log2",
            grid_axis="both",
            ylim=(0, 118),
            hide_left_spine=False,
        )
        place_dot_labels(ax, dot_pts)

    save_figure(fig, PLOT_DIR / "work_efficiency", tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot saved: {PLOT_DIR / 'work_efficiency'}")


def generate_performance_plot():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl - skipping performance plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    palette = series_palette(description_order(df))

    plot_performance_inputsize(df, palette)
    plot_performance_inputsize_theory(df, palette)
    plot_performance_inputsize_ipc(df, palette)
    plot_runtime_inputsize(df, palette)
    plot_speedup_inputsize(df, palette)
    plot_work_efficiency(df, palette)
    for oi_col, level_label in OI_LEVELS:
        plot_oi_inputsize(df, palette, oi_col, level_label)
