import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from data_io import description_order, load_history
from style import save_figure, series_palette

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as app_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / "results" / "history.jsonl"
PLOT_DIR = REPO_ROOT / "results" / "plots"

PERF_COL = "perf"
X_COL = "N"

HATCHES = ["///", "...", "xx", "\\\\", "---"]


def _perf_by_n(df_desc):
    out = {}
    for _, row in df_desc.iterrows():
        n = row[X_COL]
        v = row[PERF_COL]
        if n is not None and np.isfinite(v) and v > 0:
            out[int(n)] = float(v)
    return out


def _build_matrix(df, ref_desc):
    ref_rows = df[df["description"] == ref_desc]
    if ref_rows.empty:
        return None, None, None
    ref_perf = _perf_by_n(ref_rows)
    if not ref_perf:
        return None, None, None

    descs = [ref_desc]
    matrix = {ref_desc: {n: 1.0 for n in ref_perf}}

    for desc in description_order(df):
        if desc == ref_desc:
            continue
        perf = _perf_by_n(df[df["description"] == desc])
        if not perf:
            continue
        descs.append(desc)
        matrix[desc] = {
            n: (perf[n] / rp if (n in perf and rp > 0) else float("nan"))
            for n, rp in ref_perf.items()
        }

    n_values = sorted(ref_perf.keys())
    return descs, n_values, matrix


def _draw(
    ax, descs, n_values, matrix, palette, ref_desc, show_legend=True, n_slots=None
):
    group_w = 0.75
    bar_w = group_w / len(descs)
    x = np.arange(len(n_values))

    for i, desc in enumerate(descs):
        hatch = HATCHES[i % len(HATCHES)]
        offsets = x + (i - len(descs) / 2 + 0.5) * bar_w
        values = [matrix[desc].get(n, float("nan")) for n in n_values]

        bars = ax.bar(
            offsets,
            values,
            width=bar_w * 0.88,
            color=palette[desc],
            hatch=hatch,
            edgecolor="black",
            linewidth=0.8,
            label=desc,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if np.isfinite(val) and val > 0:
                if desc == ref_desc:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val * 1.08,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="black",
                    zorder=5,
                )

    ax.axhline(1.0, color="#333333", linewidth=1.4, linestyle="-", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in n_values], fontsize=9)
    ax.set_xlabel("Input Size", fontsize=10)

    ax.set_xlim(-0.5, (n_slots if n_slots is not None else len(n_values)) - 0.5)

    ax.set_yscale("log", base=10)
    ax.set_ylim(1e-2, 1e3)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=50)
    )
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.grid(
        True, which="major", color="lightgray", linestyle=":", linewidth=0.7, zorder=0
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    if show_legend:
        ax.legend(
            loc="upper left",
            ncol=len(descs),
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=True,
        )


MAX_GROUPS_PER_ROW = 3


def _generate(df, palette, ref_desc, ylabel, title, output_name):
    descs, n_values, matrix = _build_matrix(df, ref_desc)
    if descs is None:
        print(f"No {ref_desc} in history or no hw data - skipping {output_name}.")
        return

    chunks = [
        n_values[i : i + MAX_GROUPS_PER_ROW]
        for i in range(0, len(n_values), MAX_GROUPS_PER_ROW)
    ]
    nrows = len(chunks)
    row_w = max(8, 1.8 * MAX_GROUPS_PER_ROW * len(descs))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(row_w, 6 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for idx, (ax, chunk) in enumerate(zip(axes, chunks)):
        _draw(
            ax,
            descs,
            chunk,
            matrix,
            palette,
            ref_desc,
            show_legend=True,
            n_slots=MAX_GROUPS_PER_ROW,
        )
        ax.set_ylabel(ylabel, fontsize=10)

    axes[0].set_title(title, fontsize=12, fontweight="bold", pad=10)

    out = PLOT_DIR / output_name
    save_figure(fig, out, dpi=180)
    print(f"Plot saved: {out}")


def generate_speedup_bars():
    df = load_history(HISTORY_FILE)
    if df.empty:
        print("No data in history.jsonl - skipping speedup bar charts.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    palette = series_palette(description_order(df))

    _generate(
        df,
        palette,
        ref_desc="ezgatr",
        ylabel="Speedup over ezgatr [x]",
        title=f"Speedup vs ezgatr - {app_config.MACHINE}",
        output_name="speedup_bars_ezgatr",
    )
    _generate(
        df,
        palette,
        ref_desc="baseline",
        ylabel="Speedup over baseline [x]",
        title=f"Speedup vs baseline - {app_config.MACHINE}",
        output_name="speedup_bars_baseline",
    )
