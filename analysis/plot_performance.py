import json
import os
import sys
from pathlib import Path

import pandas as pd
from plot_style import (
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


def _cycles_to_perf(n_series, cycles_series):
    n_numeric = pd.to_numeric(n_series, errors="coerce")
    cycles_numeric = pd.to_numeric(cycles_series, errors="coerce")
    flops = n_numeric.apply(app_config.calculate_total_flops)
    out = flops / cycles_numeric
    return out.replace([float("inf"), float("-inf")], pd.NA).dropna()


def _read_jsonl(path):
    if not path.exists():
        return []

    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def load_data():
    return pd.DataFrame(row for row in _read_jsonl(HISTORY_FILE) if "data" in row)


def load_reference():
    if not REFERENCE_FILE.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    data = payload.get("data", []) if isinstance(payload, dict) else []
    return pd.DataFrame(data if isinstance(data, list) else [])


def _aligned_perf_frame(n_series, cycles_series):
    return pd.DataFrame(
        {
            "N": pd.to_numeric(n_series, errors="coerce"),
            "perf": _cycles_to_perf(n_series, cycles_series),
        }
    ).dropna()


def generate_plot():
    history_df = load_data()
    reference_df = load_reference()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = new_single_axes(figsize=(10, 6))

    plot_lines = []
    all_y = []

    if {"N", "cycles"}.issubset(reference_df.columns):
        aligned = _aligned_perf_frame(reference_df["N"], reference_df["cycles"])
        if not aligned.empty:
            plot_lines.append(
                add_series(
                    ax,
                    x=aligned["N"],
                    y=aligned["perf"],
                    label="ezgatr",
                    primary=True,
                    linestyle="--",
                    linewidth=3.2,
                    marker="o",
                )
            )
            all_y.extend(float(v) for v in aligned["perf"])

    if not history_df.empty:
        latest = history_df.drop_duplicates(subset=["description"], keep="last")
        for _, run in latest.iterrows():
            desc = str(run.get("description", "run")).strip() or "run"
            points = run.get("data", [])
            if not isinstance(points, list) or not points:
                continue

            run_df = pd.DataFrame(points)
            if not {"N", "cycles"}.issubset(run_df.columns):
                continue

            aligned = _aligned_perf_frame(run_df["N"], run_df["cycles"])
            if aligned.empty:
                continue

            plot_lines.append(
                add_series(
                    ax,
                    x=aligned["N"],
                    y=aligned["perf"],
                    label=desc,
                    linewidth=2,
                    marker="s",
                )
            )
            all_y.extend(float(v) for v in aligned["perf"])

    y_max = max(all_y) if all_y else 1.0
    style_axes(
        ax,
        title="Performance",
        y_unit_text="Performance [flops / cycle]",
        x_label="Input Size N",
        x_scale="log2",
        grid_axis="y",
        ylim=(0, max(1.0, y_max * 1.2)),
        hide_left_spine=True,
    )

    place_line_labels(
        plot_lines,
        label_adjustments=MANUAL_LABEL_ADJUSTMENTS,
    )

    output_path = PLOT_DIR / "plot_latest.pdf"
    save_figure(fig, output_path, tight_rect=(0, 0, 1.0, 1.0))
    print(f"Plot successfully generated at: {output_path}")


if __name__ == "__main__":
    generate_plot()
