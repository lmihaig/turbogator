import hashlib
from pathlib import Path

import matplotlib
import matplotlib.transforms as _mtrans
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "teal": "#0d9488",
    "magenta": "#c026d3",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "rose": "#e11d48",
    "cyan": "#0284c7",
    "gold": "#ca8a04",
    "navy": "#1e3a8a",
    "brick": "#9f1239",
    "slate": "#475569",
    "primary": "#8B0000",
    "gray_text": "0.4",
    "black": "#000000",
}

SERIES_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["teal"],
    PALETTE["magenta"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["rose"],
    PALETTE["cyan"],
    PALETTE["gold"],
    PALETTE["navy"],
    PALETTE["brick"],
    PALETTE["slate"],
]


PRIMARY_DESC = "ezgatr"


def _normalize_desc(desc):
    return "".join(str(desc).lower().split())


def series_palette(descriptions, primary=PRIMARY_DESC):
    primary_norm = _normalize_desc(primary)
    out = {}
    used_colors = set()

    for d in descriptions:
        if d in out:
            continue

        nd = _normalize_desc(d)

        if nd == primary_norm:
            out[d] = PALETTE["primary"]
            continue

        h = int.from_bytes(hashlib.md5(nd.encode()).digest()[:4], "big")
        base_idx = h % len(SERIES_COLORS)
        color = SERIES_COLORS[base_idx]

        if color in used_colors and len(used_colors) < len(SERIES_COLORS):
            for offset in range(1, len(SERIES_COLORS)):
                probe_idx = (base_idx + offset) % len(SERIES_COLORS)
                if SERIES_COLORS[probe_idx] not in used_colors:
                    color = SERIES_COLORS[probe_idx]
                    break

        out[d] = color
        used_colors.add(color)

    return out


DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 180


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


def _text_dims(ax, renderer, text, fontsize, fontweight="normal"):
    t = ax.text(
        0.5,
        0.5,
        text,
        fontsize=fontsize,
        fontweight=fontweight,
        ha="center",
        va="center",
        alpha=0.0,
        transform=ax.transAxes,
    )
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.x1 - bb.x0, bb.y1 - bb.y0


def _label_bbox(cx, cy, tw, th, pad=2):
    return _mtrans.Bbox.from_extents(
        cx - tw / 2 - pad,
        cy - th / 2 - pad,
        cx + tw / 2 + pad,
        cy + th / 2 + pad,
    )


def place_line_labels(lines, *, label_adjustments=None):
    candidates = []
    for line in lines:
        lbl = str(line.get_label()).strip()
        if not lbl or lbl.startswith("_"):
            continue
        xa = np.atleast_1d(np.asarray(line.get_xdata(orig=False), dtype=float))
        ya = np.atleast_1d(np.asarray(line.get_ydata(orig=False), dtype=float))
        ok = np.isfinite(xa) & np.isfinite(ya)
        if ok.sum() == 0:
            continue
        candidates.append((line, xa[ok], ya[ok], lbl, str(line.get_color())))
    if not candidates:
        return

    ax = candidates[0][0].axes
    if ax is None:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_bbox = ax.bbox
    line_paths = [
        ln.get_path().transformed(ln.get_transform()) for ln, *_ in candidates
    ]
    placed_boxes = []

    n_lines = len(candidates)
    placed_centers = []

    # nudges

    EDGE_GAPS = [5, 9, 14, 20, 28, 38]
    DX_FRACS = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]

    for i, (line, xa, ya, lbl, color) in enumerate(candidates):
        tw, th = _text_dims(ax, renderer, lbl, 10, "bold")

        adj_dx, adj_dy = 0.0, 0.0
        if label_adjustments and lbl in label_adjustments:
            adj_dx, adj_dy = label_adjustments[lbl]

        pts = ax.transData.transform(np.column_stack([xa, ya]))
        n = len(xa)
        frac = 0.5 if n_lines == 1 else 0.18 + 0.64 * i / (n_lines - 1)
        pref = frac * (n - 1)
        idx_order = sorted(range(n), key=lambda j: abs(j - pref))

        offsets = []
        for gap in EDGE_GAPS:
            cy = gap + th / 2
            for fx in DX_FRACS:
                dx = fx * tw
                offsets.append((cy, dx))
                offsets.append((-cy, dx))
        offsets.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)

        x_sep = tw + 18

        best_score, best_result = None, None
        found = False

        for idx in idx_order:
            if found:
                break
            xd, yd = pts[idx]
            for cy_off, cx_off in offsets:
                tx = xd + cx_off + adj_dx
                ty = yd + cy_off + adj_dy
                bb = _label_bbox(tx, ty, tw, th)

                inside = (
                    bb.x0 >= plot_bbox.x0
                    and bb.x1 <= plot_bbox.x1
                    and bb.y0 >= plot_bbox.y0
                    and bb.y1 <= plot_bbox.y1
                )
                ln_hits = sum(
                    1
                    for j, p in enumerate(line_paths)
                    if j != i and p.intersects_bbox(bb, filled=False)
                )
                lb_hits = sum(1 for b in placed_boxes if b.overlaps(bb))

                x_pen = sum(max(0.0, x_sep - abs(tx - px)) for px in placed_centers)
                dist = (cx_off**2 + cy_off**2) ** 0.5
                prox = dist + 0.05 * dist * dist
                score = (
                    prox
                    + 1.0 * x_pen
                    + 800 * ln_hits
                    + 1200 * lb_hits
                    + (5000 if not inside else 0)
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_result = (tx, ty, bb)

                if inside and ln_hits == 0 and lb_hits == 0 and x_pen == 0.0:
                    found = True
                    break

        if best_result is not None:
            tx, ty, bb = best_result
            td = ax.transData.inverted().transform((tx, ty))
            ax.text(
                float(td[0]),
                float(td[1]),
                lbl,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
                zorder=5,
            )
            placed_boxes.append(bb)
            placed_centers.append(tx)


def place_dot_labels(
    ax, points, *, fontsize=9, label_adjustments=None, marker_sizes=None
):

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_bbox = ax.bbox

    dot_boxes = []
    for i, (lbl, x_data, y_data, color) in enumerate(points):
        s = marker_sizes[i] if marker_sizes is not None else 150.0
        half_px = (float(s) ** 0.5) / 2.0 * fig.dpi / 72.0
        dcx, dcy = ax.transData.transform((x_data, y_data))
        dot_boxes.append(_label_bbox(dcx, dcy, 2 * half_px, 2 * half_px, pad=2))

    placed_boxes = []
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    dirs = [(np.cos(a), np.sin(a)) for a in angles]
    MAGNITUDES = [28, 38, 50, 65, 84, 108]

    offsets_base = sorted(
        [(mag * cx, mag * cy) for mag in MAGNITUDES for cx, cy in dirs],
        key=lambda p: p[0] ** 2 + p[1] ** 2,
    )

    for lbl, x_data, y_data, color in points:
        tw, th = _text_dims(ax, renderer, lbl, fontsize, "bold")

        adj_dx, adj_dy = 0.0, 0.0
        if label_adjustments and lbl in label_adjustments:
            adj_dx, adj_dy = label_adjustments[lbl]

        xd, yd = ax.transData.transform((x_data, y_data))

        best_score, best_result = None, None
        for dx_px, dy_px in offsets_base:
            tx = xd + dx_px + adj_dx
            ty = yd + dy_px + adj_dy
            bb = _label_bbox(tx, ty, tw, th)

            inside = (
                bb.x0 >= plot_bbox.x0
                and bb.x1 <= plot_bbox.x1
                and bb.y0 >= plot_bbox.y0
                and bb.y1 <= plot_bbox.y1
            )
            lb_hits = sum(1 for b in placed_boxes if b.overlaps(bb))
            dot_hits = sum(1 for b in dot_boxes if b.overlaps(bb))
            dist = (dx_px**2 + dy_px**2) ** 0.5
            score = (
                dist + 1500 * lb_hits + 1500 * dot_hits + (5000 if not inside else 0)
            )

            if best_score is None or score < best_score:
                best_score = score
                td = ax.transData.inverted().transform((tx, ty))
                best_result = (float(td[0]), float(td[1]), bb)

            if inside and lb_hits == 0 and dot_hits == 0:
                break

        if best_result is not None:
            tx, ty, bb = best_result
            ax.text(
                tx,
                ty,
                lbl,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=color,
                zorder=5,
            )
            placed_boxes.append(bb)


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
