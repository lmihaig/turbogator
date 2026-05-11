import os
import sys
from pathlib import Path

import pandas as pd
from data_io import load_history_dataframe, load_reference_dataframe
from style import (
    add_series,
    new_single_axes,
    place_line_labels,
    save_figure,
    style_axes,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
REFERENCE_FILE = REPO_ROOT / "results" / "reference" / "metrics.json"
PLOT_DIR = REPO_ROOT / "results" / "plots"

# move overlapping labels: "exact label string": (x_offset, y_offset)
# offsets are roughly % of the axis span (+ is right/up, - is left/down).
MANUAL_LABEL_ADJUSTMENTS = {
    "test2 (scalar)": (-12.0, 1.2),
}


def _frame_from_points(points):
    df = pd.DataFrame(points)
    x = df["X"]
    size = x.apply(lambda v: float(v["T"]) * float(v["C_in"]))
    ar = x.apply(lambda v: float(v["T"]) / float(v["C_in"]))
    cycles = df["cycles"].astype(float)
    flops = x.apply(
        lambda v: app_config.calculate_total_flops(v["B"], v["T"], v["C_in"], v["D"])
    )
    perf = flops / cycles
    frame = pd.DataFrame({"size": size, "ar": ar, "cycles": cycles, "perf": perf})
    return frame.sort_values("size", kind="mergesort")


def plot_performance_inputsize(history_df, reference_df):
    fig, ax = new_single_axes(figsize=(10, 6))
    plot_lines = []
    all_y = []

    if not reference_df.empty:
        ref = _frame_from_points(reference_df.to_dict("records"))
        plot_lines.append(
            add_series(
                ax,
                x=ref["size"],
                y=ref["perf"],
                label="ezgatr",
                primary=True,
                linestyle="--",
                linewidth=3.2,
                marker="o",
            )
        )
        all_y.extend(float(v) for v in ref["perf"])

    if not history_df.empty:
        latest = history_df.drop_duplicates(subset=["description"], keep="last")
        for _, run in latest.iterrows():
            desc = str(run["description"]).strip() or "run"
            frame = _frame_from_points(run["data"])
            plot_lines.append(
                add_series(
                    ax,
                    x=frame["size"],
                    y=frame["perf"],
                    label=desc,
                    linewidth=2,
                    marker="s",
                )
            )
            all_y.extend(float(v) for v in frame["perf"])

    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Performance: {app_config.MACHINE}",
        y_unit_text="Performance [flops / cycle]",
        x_label="Input Size n = log2(T*C_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )

    place_line_labels(
        plot_lines,
        label_adjustments=MANUAL_LABEL_ADJUSTMENTS,
    )

    output_path = PLOT_DIR / "performance_inputsize"
    save_figure(fig, output_path, tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot successfully generated at: {output_path}")


def plot_performance_ar(history_df, reference_df):
    fig, ax = new_single_axes(figsize=(10, 6))
    plot_lines = []
    all_y = []

    if not reference_df.empty:
        ref = _frame_from_points(reference_df.to_dict("records"))
        plot_lines.append(
            add_series(
                ax,
                x=ref["ar"],
                y=ref["perf"],
                label="ezgatr",
                primary=True,
                linestyle="--",
                linewidth=3.2,
                marker="o",
            )
        )
        all_y.extend(float(v) for v in ref["perf"])

    if not history_df.empty:
        latest = history_df.drop_duplicates(subset=["description"], keep="last")
        for _, run in latest.iterrows():
            desc = str(run["description"]).strip() or "run"
            frame = _frame_from_points(run["data"])
            plot_lines.append(
                add_series(
                    ax,
                    x=frame["ar"],
                    y=frame["perf"],
                    label=desc,
                    linewidth=2,
                    marker="s",
                )
            )
            all_y.extend(float(v) for v in frame["perf"])

    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Performance (Aspect Ratio): {app_config.MACHINE}",
        y_unit_text="Performance [flops / cycle]",
        x_label="Tensor Aspect Ratio (T/C_in)",
        x_scale="linear",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )

    place_line_labels(
        plot_lines,
        label_adjustments=MANUAL_LABEL_ADJUSTMENTS,
    )

    output_path = PLOT_DIR / "performance_ar"
    save_figure(fig, output_path, tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot successfully generated at: {output_path}")


def plot_runtime_inputsize(history_df, reference_df):
    fig, ax = new_single_axes(figsize=(10, 6))
    plot_lines = []
    all_y = []

    if not reference_df.empty:
        ref = _frame_from_points(reference_df.to_dict("records"))
        plot_lines.append(
            add_series(
                ax,
                x=ref["size"],
                y=ref["cycles"] / 1e9,
                label="ezgatr",
                primary=True,
                linestyle="--",
                linewidth=3.2,
                marker="o",
            )
        )
        all_y.extend(float(v) for v in (ref["cycles"] / 1e9))

    if not history_df.empty:
        latest = history_df.drop_duplicates(subset=["description"], keep="last")
        for _, run in latest.iterrows():
            desc = str(run["description"]).strip() or "run"
            frame = _frame_from_points(run["data"])
            plot_lines.append(
                add_series(
                    ax,
                    x=frame["size"],
                    y=frame["cycles"] / 1e9,
                    label=desc,
                    linewidth=2,
                    marker="s",
                )
            )
            all_y.extend(float(v) for v in (frame["cycles"] / 1e9))

    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title=f"Runtime: {app_config.MACHINE}",
        y_unit_text="Time [Gcycles]",
        x_label="Input Size (T*C_in)",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )

    place_line_labels(
        plot_lines,
        label_adjustments=MANUAL_LABEL_ADJUSTMENTS,
    )

    output_path = PLOT_DIR / "runtime_inputsize"
    save_figure(fig, output_path, tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot successfully generated at: {output_path}")


def generate_performance_plot():
    history_df = load_history_dataframe(HISTORY_FILE)
    reference_df = load_reference_dataframe(REFERENCE_FILE)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_performance_inputsize(history_df, reference_df)
    plot_performance_ar(history_df, reference_df)
    plot_runtime_inputsize(history_df, reference_df)


if __name__ == "__main__":
    generate_performance_plot()
