import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style import PALETTE, new_single_axes, save_figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REPO_ROOT = Path(__file__).resolve().parents[1]
JUNK_RESULTS = REPO_ROOT / "junk" / "results"
PLOT_DIR = REPO_ROOT / "results" / "plots"


def _to_float(token):
    t = token.replace("*", "").strip()
    if not t:
        return float("nan")
    if t.lower() in {"inf", "nan"}:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _annotate_heatmap(ax, data):
    for (y, x), val in np.ndenumerate(data):
        label = "n/a" if not np.isfinite(val) else f"{val:.2f}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
        )


def _best_value(data, x_vals, y_vals):
    mask = np.isfinite(data)
    if not mask.any():
        return None
    idx = np.nanargmax(data)
    y_idx, x_idx = np.unravel_index(idx, data.shape)
    return data[y_idx, x_idx], x_vals[x_idx], y_vals[y_idx]


def _parse_mr_nr_blocks(lines, start_idx):
    header = lines[start_idx].split()
    x_vals = [int(v) for v in header[1:] if v.isdigit()]

    y_vals = []
    rows = []
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            break
        if line.startswith("---") or line.lower().startswith("best"):
            break
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        y = int(parts[0])
        values = [_to_float(t) for t in parts[1 : 1 + len(x_vals)]]
        if len(values) == len(x_vals):
            y_vals.append(y)
            rows.append(values)
        i += 1

    return x_vals, y_vals, rows


def _parse_av_tile(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    results = []
    i = 0
    while i < len(lines):
        m = re.match(r"---\s+N=(\d+)", lines[i])
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        i += 1
        while i < len(lines) and "MR\\NR" not in lines[i]:
            i += 1
        if i >= len(lines):
            break
        x_vals, y_vals, rows = _parse_mr_nr_blocks(lines, i)
        if rows:
            results.append((n, x_vals, y_vals, np.array(rows, dtype=float)))
        i += 1
    return results


def _parse_gemm_block(path):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "MR\\NR" in line:
            x_vals, y_vals, rows = _parse_mr_nr_blocks(lines, i)
            if rows:
                return x_vals, y_vals, np.array(rows, dtype=float)
            break
    return None


def _parse_bt_sweep(path):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    values = {}
    current_n = None
    for line in lines:
        m = re.match(r"^N=\s*(\d+)", line)
        if m:
            current_n = int(m.group(1))
            continue
        m = re.search(r"flash\s+BT=(\d+)\s+.*\s+([0-9.]+)$", line)
        if m and current_n is not None:
            bt = int(m.group(1))
            values[(current_n, bt)] = float(m.group(2))

    if not values:
        return None
    n_vals = sorted({n for n, _ in values.keys()})
    bt_vals = sorted({bt for _, bt in values.keys()})
    data = np.full((len(n_vals), len(bt_vals)), np.nan, dtype=float)
    for (n, bt), val in values.items():
        y = n_vals.index(n)
        x = bt_vals.index(bt)
        data[y, x] = val
    return bt_vals, n_vals, data


def _parse_panel_sweep(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    results = []
    section = None
    values = {}

    def flush():
        if section is None or not values:
            return
        n_vals = sorted({n for n, _ in values.keys()})
        nc_vals = sorted({nc for _, nc in values.keys()})
        data = np.full((len(n_vals), len(nc_vals)), np.nan, dtype=float)
        for (n, nc), val in values.items():
            y = n_vals.index(n)
            x = nc_vals.index(nc)
            data[y, x] = val
        results.append((section, nc_vals, n_vals, data))

    for line in lines:
        m = re.match(r"^---\s+([^\s]+)", line)
        if m:
            flush()
            section = m.group(1)
            values = {}
            continue
        m = re.match(
            r"^\s*(\d+)\s+(\d+)\s+\|\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)",
            line,
        )
        if m:
            n = int(m.group(1))
            nc = int(m.group(3))
            val = float(m.group(7))
            values[(n, nc)] = val

    flush()
    return results


import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style import PALETTE, new_single_axes, save_figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REPO_ROOT = Path(__file__).resolve().parents[1]
JUNK_RESULTS = REPO_ROOT / "junk" / "results"
PLOT_DIR = REPO_ROOT / "results" / "plots"


def _to_float(token):
    t = token.replace("*", "").strip()
    if not t:
        return float("nan")
    if t.lower() in {"inf", "nan"}:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _annotate_heatmap(ax, data):
    for (y, x), val in np.ndenumerate(data):
        label = "n/a" if not np.isfinite(val) else f"{val:.2f}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
        )


def _best_value(data, x_vals, y_vals):
    mask = np.isfinite(data)
    if not mask.any():
        return None
    idx = np.nanargmax(data)
    y_idx, x_idx = np.unravel_index(idx, data.shape)
    return data[y_idx, x_idx], x_vals[x_idx], y_vals[y_idx]


def _parse_mr_nr_blocks(lines, start_idx):
    header = lines[start_idx].split()
    x_vals = [int(v) for v in header[1:] if v.isdigit()]

    y_vals = []
    rows = []
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            break
        if line.startswith("---") or line.lower().startswith("best"):
            break
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        y = int(parts[0])
        values = [_to_float(t) for t in parts[1 : 1 + len(x_vals)]]
        if len(values) == len(x_vals):
            y_vals.append(y)
            rows.append(values)
        i += 1

    return x_vals, y_vals, rows


def _parse_av_tile(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    results = []
    i = 0
    while i < len(lines):
        m = re.match(r"---\s+N=(\d+)", lines[i])
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        i += 1
        while i < len(lines) and "MR\\NR" not in lines[i]:
            i += 1
        if i >= len(lines):
            break
        x_vals, y_vals, rows = _parse_mr_nr_blocks(lines, i)
        if rows:
            results.append((n, x_vals, y_vals, np.array(rows, dtype=float)))
        i += 1
    return results


def _parse_gemm_block(path):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "MR\\NR" in line:
            x_vals, y_vals, rows = _parse_mr_nr_blocks(lines, i)
            if rows:
                return x_vals, y_vals, np.array(rows, dtype=float)
            break
    return None


def _parse_bt_sweep(path):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    values = {}
    current_n = None
    for line in lines:
        m = re.match(r"^N=\s*(\d+)", line)
        if m:
            current_n = int(m.group(1))
            continue
        m = re.search(r"flash\s+BT=(\d+)\s+.*\s+([0-9.]+)$", line)
        if m and current_n is not None:
            bt = int(m.group(1))
            values[(current_n, bt)] = float(m.group(2))

    if not values:
        return None
    n_vals = sorted({n for n, _ in values.keys()})
    bt_vals = sorted({bt for _, bt in values.keys()})
    data = np.full((len(n_vals), len(bt_vals)), np.nan, dtype=float)
    for (n, bt), val in values.items():
        y = n_vals.index(n)
        x = bt_vals.index(bt)
        data[y, x] = val
    return bt_vals, n_vals, data


def _parse_panel_sweep(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    results = []
    section = None
    values = {}

    def flush():
        if section is None or not values:
            return
        n_vals = sorted({n for n, _ in values.keys()})
        nc_vals = sorted({nc for _, nc in values.keys()})
        data = np.full((len(n_vals), len(nc_vals)), np.nan, dtype=float)
        for (n, nc), val in values.items():
            y = n_vals.index(n)
            x = nc_vals.index(nc)
            data[y, x] = val
        results.append((section, nc_vals, n_vals, data))

    for line in lines:
        m = re.match(r"^---\s+([^\s]+)", line)
        if m:
            flush()
            section = m.group(1)
            values = {}
            continue
        m = re.match(
            r"^\s*(\d+)\s+(\d+)\s+\|\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)",
            line,
        )
        if m:
            n = int(m.group(1))
            nc = int(m.group(3))
            val = float(m.group(7))
            values[(n, nc)] = val

    flush()
    return results


def _plot_heatmap(
    *,
    title,
    x_vals,
    y_vals,
    data,
    x_label,
    y_label,
    x_name,
    y_name,
    metric_name,
    output_name,
):
    if data.size == 0:
        return

    data = np.asarray(data, dtype=float)
    masked = np.ma.masked_invalid(data)
    if not np.isfinite(data).any():
        return

    fig, ax = new_single_axes(figsize=(9.5, 7.5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d1d5db")

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Polished, centered title
    ax.set_title(title, fontsize=14, fontweight="bold", loc="center", pad=15)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=10)
    ax.set_yticklabels([str(v) for v in y_vals], fontsize=10)

    # Remove all axis lines, ticks, and grid overlays completely
    ax.tick_params(which="both", bottom=False, left=False, top=False, right=False)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, fontsize=10)

    _annotate_heatmap(ax, data)

    # Hide the border spine
    for spine in ax.spines.values():
        spine.set_visible(False)

    best = _best_value(data, x_vals, y_vals)
    if best is not None:
        best_val, best_x, best_y = best

        # Anchor to `ax.transAxes` instead of `fig` to center under the heatmap (ignoring colorbar width)
        ax.text(
            0.5,
            -0.14,
            f"Best {metric_name} performance: {x_name} = {best_x}, {y_name} = {best_y} ({best_val:.2f})",
            transform=ax.transAxes,
            fontsize=10,
            color=PALETTE.get("gray_text", "0.4"),
            ha="center",
            va="top",
        )

    # Ensure it writes directly to PLOT_DIR with the heatmap_ prefix
    name = (
        output_name if output_name.startswith("heatmap_") else f"heatmap_{output_name}"
    )
    save_figure(fig, PLOT_DIR / name, tight_rect=(0, 0.05, 1.0, 0.95))
    print(f"Plot saved: {PLOT_DIR / name}")


def generate_heatmaps():
    if not JUNK_RESULTS.exists():
        print("No junk/results folder - skipping heatmaps.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    av_path = JUNK_RESULTS / "attn_av_gemm_tile_sweep" / "run.log"
    for n, x_vals, y_vals, data in _parse_av_tile(av_path):
        _plot_heatmap(
            title=f"Attention AV Tile Sweep (N={n})",
            x_vals=x_vals,
            y_vals=y_vals,
            data=data,
            x_label="NR",
            y_label="MR",
            x_name="NR",
            y_name="MR",
            metric_name="FLOPs/cycle",
            output_name=f"av_tile_sweep_n{n}",
        )

    gemm_path = JUNK_RESULTS / "attn_gemm_block_sweep" / "run.log"
    parsed = _parse_gemm_block(gemm_path)
    if parsed is not None:
        x_vals, y_vals, data = parsed
        _plot_heatmap(
            title="GEMM Register Tile Sweep",
            x_vals=x_vals,
            y_vals=y_vals,
            data=data,
            x_label="NR",
            y_label="MR",
            x_name="NR",
            y_name="MR",
            metric_name="FLOPs/cycle",
            output_name="gemm_register_tile_sweep",
        )

    bt_path = JUNK_RESULTS / "attn_bt_sweep" / "run.log"
    parsed = _parse_bt_sweep(bt_path)
    if parsed is not None:
        bt_vals, n_vals, data = parsed
        _plot_heatmap(
            title="Flash Attention Block Size Sweep",
            x_vals=bt_vals,
            y_vals=n_vals,
            data=data,
            x_label="BT",
            y_label="N",
            x_name="BT",
            y_name="N",
            metric_name="FLOPs/cycle",
            output_name="flash_bt_sweep",
        )

    panel_path = JUNK_RESULTS / "equi_linear_oc_panel_sweep" / "run.log"
    for section, nc_vals, n_vals, data in _parse_panel_sweep(panel_path):
        _plot_heatmap(
            title=f"Equi-Linear Panel Sweep ({section.title()})",
            x_vals=nc_vals,
            y_vals=n_vals,
            data=data,
            x_label="NC",
            y_label="N",
            x_name="NC",
            y_name="N",
            metric_name="FLOPs/cycle",
            output_name=f"equi_linear_panel_{section}",
        )
