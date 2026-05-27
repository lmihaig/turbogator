import hashlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "green": "#4e8f7a",
    "orange": "#b87a3f",
    "purple": "#7a74a6",
    "magenta": "#b06d8f",
    "blue": "#5c85a6",
    "red": "#b24a4a",
    "teal": "#3f8a8a",
    "olive": "#8a8a3f",
    "indigo": "#5a5aa6",
    "salmon": "#cc7a6a",
    "primary": "#8B0000",
    "gray_text": "0.4",
}

SERIES_COLORS = [
    PALETTE["blue"],
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["magenta"],
    PALETTE["red"],
    PALETTE["teal"],
    PALETTE["olive"],
    PALETTE["indigo"],
    PALETTE["salmon"],
]

PRIMARY_DESC = "ezgatr"


def _normalize_desc(desc):
    return "".join(str(desc).lower().split())


def series_palette(descriptions, primary=PRIMARY_DESC):
    primary_norm = _normalize_desc(primary)
    out = {}
    for d in descriptions:
        if d in out:
            continue
        nd = _normalize_desc(d)
        if nd == primary_norm:
            out[d] = PALETTE["primary"]
        else:
            h = int.from_bytes(hashlib.md5(nd.encode()).digest()[:4], "big")
            out[d] = SERIES_COLORS[h % len(SERIES_COLORS)]
    return out


DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 180
_LABEL_BASE_FRACS = [0.34, 0.42, 0.50, 0.58, 0.66, 0.74]
_LABEL_Y_TRIALS = [1.0, -1.0, 1.8, -1.8, 2.6, -2.6]


def use_style(overrides=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Helvetica", "Gill Sans MT", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "axes.facecolor": "#efefef",
        "figure.facecolor": "white",
        "axes.axisbelow": True,
    }
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def new_figure(
    *,
    nrows=1,
    ncols=1,
    figsize=DEFAULT_FIGSIZE,
    **kwargs,
):
    use_style()
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def new_single_axes(*, figsize=DEFAULT_FIGSIZE, **kwargs):
    use_style()
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def style_axes(
    ax,
    *,
    title,
    y_unit_text,
    x_label="",
    x_scale="linear",
    x_ticks=None,
    x_tick_labels=None,
    xlim=None,
    ylim=None,
    grid_axis="y",
    grid_which="major",
    hide_left_spine=True,
):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(
        0.0,
        1.0,
        y_unit_text,
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gray_text"],
        ha="left",
        va="bottom",
    )
    ax.set_ylabel("")
    ax.set_xlabel(x_label)
    ax.set_axisbelow(True)

    if x_scale == "log2":
        ax.set_xscale("log", base=2)
    elif x_scale != "linear":
        raise ValueError(f"Unsupported x_scale: {x_scale}")

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(
        True,
        which=grid_which,
        axis=grid_axis,
        linewidth=1.0,
        color="white",
        alpha=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)


def _next_series_color(ax):
    return SERIES_COLORS[len(ax.get_lines()) % len(SERIES_COLORS)]


def add_series(
    ax,
    *,
    x,
    y,
    label,
    primary=False,
    color=None,
    linewidth=None,
    marker=None,
    alpha=None,
    zorder=None,
    **kwargs,
):
    line_color = (
        color
        if color is not None
        else (PALETTE["primary"] if primary else _next_series_color(ax))
    )
    line_width = linewidth if linewidth is not None else (3.2 if primary else 2.0)
    line_marker = marker if marker is not None else ("o" if primary else "s")
    line_alpha = alpha if alpha is not None else (1.0 if primary else 0.95)
    line_zorder = zorder if zorder is not None else (4 if primary else 3)

    (line,) = ax.plot(
        x,
        y,
        label=label,
        color=line_color,
        linewidth=line_width,
        marker=line_marker,
        alpha=line_alpha,
        zorder=line_zorder,
        **kwargs,
    )
    return line


def place_line_labels(
    lines,
    *,
    label_adjustments=None,
):
    candidates = []
    for line in lines:
        label = str(line.get_label()).strip()
        if not label or label.startswith("_"):
            continue
        x_arr = np.atleast_1d(np.asarray(line.get_xdata(orig=False), dtype=float))
        y_arr = np.atleast_1d(np.asarray(line.get_ydata(orig=False), dtype=float))
        if x_arr.size == 0 or x_arr.size != y_arr.size:
            continue
        candidates.append((line, x_arr, y_arr, label, str(line.get_color())))

    if not candidates:
        return

    ax = candidates[0][0].axes
    if ax is None:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    line_paths = [
        ln.get_path().transformed(ln.get_transform()) for ln, *_ in candidates
    ]
    placed_boxes = []

    x_left, x_right = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_jitter = max((y_max - y_min) * 0.015, 1e-6)
    y_step = max((y_max - y_min) * 0.01, 1e-12)

    x_scale = ax.get_xscale()
    if x_scale == "log":
        lo = max(min(x_left, x_right), 1e-12)
        hi = max(max(x_left, x_right), lo * 1.0000001)
        x_step = max((np.log2(hi) - np.log2(lo)) * 0.01, 1e-12)
    else:
        x_step = max(abs(x_right - x_left) * 0.01, 1e-12)

    for i, (_, x_target, y_target, label, color) in enumerate(candidates):
        n = int(x_target.size)

        x_units = 0.0
        y_units = 0.0
        if label_adjustments is not None and label in label_adjustments:
            x_units, y_units = label_adjustments[label]

        # Each line gets a different preferred middle anchor by rotating fractions.
        shift = i % len(_LABEL_BASE_FRACS)
        fracs = _LABEL_BASE_FRACS[shift:] + _LABEL_BASE_FRACS[:shift]

        idx_candidates = []
        seen_idx = set()
        for frac in fracs:
            idx = int(round((n - 1) * frac))
            idx = max(1, min(n - 2, idx)) if n > 2 else max(0, min(n - 1, idx))
            if idx not in seen_idx:
                seen_idx.add(idx)
                idx_candidates.append(idx)

        best = None
        placed = False

        for idx in idx_candidates:
            x_pos = float(x_target[idx])
            y_pos = float(y_target[idx])

            if x_scale == "log":
                x_base = np.log2(max(x_pos, 1e-12)) + (float(x_units) * x_step)
                x_text = 2**x_base
            else:
                x_text = x_pos + (float(x_units) * x_step)

            for mult in _LABEL_Y_TRIALS:
                y_text = y_pos + (mult * y_jitter) + (float(y_units) * y_step)

                ghost = ax.text(
                    x_text,
                    y_text,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    alpha=0.0,
                )
                bbox = ghost.get_window_extent(renderer=renderer)
                ghost.remove()
                padded = bbox.from_extents(
                    bbox.x0 - 1.0, bbox.y0 - 1.0, bbox.x1 + 1.0, bbox.y1 + 1.0
                )

                in_axes = (
                    padded.x0 >= axes_bbox.x0
                    and padded.x1 <= axes_bbox.x1
                    and padded.y0 >= axes_bbox.y0
                    and padded.y1 <= axes_bbox.y1
                )
                line_overlaps = sum(
                    1
                    for path in line_paths
                    if path.intersects_bbox(padded, filled=False)
                )
                label_overlaps = sum(1 for box in placed_boxes if box.overlaps(padded))
                score = (1000 * line_overlaps) + (1200 * label_overlaps) + abs(mult)
                if not in_axes:
                    score += 5000

                if best is None or score < best[2]:
                    best = (x_text, y_text, score, padded)

                if in_axes and line_overlaps == 0 and label_overlaps == 0:
                    ax.text(
                        x_text,
                        y_text,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        color=color,
                        zorder=4,
                    )
                    placed_boxes.append(padded)
                    placed = True
                    break

            if placed:
                break

        if not placed and best is not None:
            x_text, y_text, _, padded = best
            ax.text(
                x_text,
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
                zorder=4,
            )
            placed_boxes.append(padded)


def annotate_bar_values(
    ax,
    bars,
    *,
    decimals=2,
    y_offset=None,
    fontsize=9,
    color="#333333",
):
    heights = [bar.get_height() for bar in bars]
    max_height = max(heights) if heights else 0.0
    offset = y_offset if y_offset is not None else max_height * 0.02

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


DEFAULT_SAVE_FMTS = ["png", "svg"]


def save_figure(
    fig,
    output_path,
    *,
    dpi=DEFAULT_DPI,
    tight_rect=(0, 0, 1.0, 0.95),
):
    output = Path(output_path)
    base_name = output.with_suffix("").name
    parent_dir = output.parent

    fig.tight_layout(rect=tight_rect)

    saved = []
    for f in DEFAULT_SAVE_FMTS:
        fmt_dir = parent_dir / f
        fmt_dir.mkdir(parents=True, exist_ok=True)

        out = fmt_dir / f"{base_name}.{f}"

        fig.savefig(out, dpi=dpi, bbox_inches="tight", format=f)
        saved.append(out)

    plt.close(fig)
    return saved[0] if len(saved) == 1 else saved
