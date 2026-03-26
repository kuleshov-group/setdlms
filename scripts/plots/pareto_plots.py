#
import math
import re

import numpy as np
import plotly.graph_objects as go

avg_block_size = 4
simple_version = True
simple_s = 4  # which block size for simple version: 4 (S≤8) or 16 (S≤32)
xaxis_key = "speed_x"

filename = (
    f"line_plot_block{avg_block_size}_{xaxis_key}_simple{simple_version}"
    + (f"_s{simple_s}" if simple_version else "")
    + ".png"
)
show_xerr = False

save_pdf = True
pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"

if xaxis_key == "x":
    xaxis_min = 0.8
    xaxis_max = 4.2
else:
    xaxis_min = 30
    xaxis_max = 170
if simple_version and xaxis_key == "speed_x":
    xaxis_min = 50 if simple_s == 4 else 65  # after dropping lowest speed point
    xaxis_max = 130 if simple_s == 4 else 165  # S=4 ~122, S=16 ~160
AR_COLOR = "#636EFA"
FONT_FAMILY = "DM Sans"

legend_font_size = 14

xaxis_std_key = xaxis_key + "_std"

_S_RE = re.compile(r"S\s*=\s*(\d+)")
FIXED_COLOR_BY_S: dict[int, str] = {
    4: "#ff7f0e",
    16: "#1f77b4",
}


def _extract_s(name: str) -> int | None:
    m = _S_RE.search(name)
    return int(m.group(1)) if m else None


def _is_block_diff(name: str) -> bool:
    n = name.lower()
    return ("block diff" in n) and ("set" not in n)


def _is_soft_block_diff(name: str) -> bool:
    return "set" in name.lower()


def _unified_color_by_s(s_values: list[int]) -> dict[int, str]:
    if not s_values:
        return {}
    s_sorted = sorted(set(s_values))
    palette = [
        "#ff7f0e",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    out: dict[int, str] = {}
    for i, s in enumerate(s_sorted):
        out[s] = FIXED_COLOR_BY_S.get(s, palette[i % len(palette)])
    return out


def _paper_aspect(fig: go.Figure) -> float:
    m = fig.layout.margin
    pw = float(fig.layout.width - (m.l + m.r))
    ph = float(fig.layout.height - (m.t + m.b))
    return ph / max(pw, 1e-9)


ar_dict = {
    "name": "AR",
    "x": np.array([1.0]),
    "x_std": np.array([0.0]),
    "speed_x": np.array([55.52]),
    "speed_x_std": np.array([0.25]),
    "y": np.array([80.21]),
}


def _legend_grouping(name: str, *, simple: bool = False) -> tuple[str | None, str, int]:
    if name == "AR":
        return ("AR", "AR", 0)
    s = _extract_s(name)
    if s is None:
        return (None, name, 0)
    lg = f"Block size {s}"
    if _is_block_diff(name):
        return (lg, "Block Diffusion" if simple else f"Block Diffusion (S={int(s)})", 0)
    if _is_soft_block_diff(name):
        return (
            lg,
            (
                "Set Diffusion (Ours)"
                if simple
                else f"Ours: Set Diffusion (S ≤ {int(s * 2)})"
            ),
            1,
        )
    return (lg, name, 2)


def _add_custom_grouped_legend(
    fig: go.Figure,
    *,
    groups_: list[dict],
    block_color_by_s: dict[int, str],
    soft_color_by_s: dict[int, str],
    simple: bool = False,
):
    s_present = sorted(
        {s for s in (_extract_s(g["name"]) for g in groups_) if s is not None}
    )
    has_ar = any(g.get("name") == "AR" for g in groups_)

    if not s_present and not has_ar:
        return

    if simple:
        block_color_by_s = {s: SIMPLE_BLOCK_COLOR for s in s_present}
        soft_color_by_s = {s: SIMPLE_SOFT_COLOR for s in s_present}

    legend_text_size = 22 if simple else legend_font_size  # larger in simple version

    def _star_path(cx, cy, r_outer, r_inner, aspect, n=5):
        ry_o = r_outer / max(aspect, 1e-9)
        ry_i = r_inner / max(aspect, 1e-9)
        pts = []
        for i in range(2 * n):
            ang = -math.pi * 3 / 2 + i * (math.pi / n)
            rx, ry = (r_outer, ry_o) if i % 2 == 0 else (r_inner, ry_i)
            pts.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))
        return "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"

    def _diamond_path(cx, cy, rx, ry, aspect):
        ry_adj = ry / max(aspect, 1e-9)
        return (
            f"M {cx},{cy + ry_adj} L {cx + rx},{cy} L {cx},{cy - ry_adj} "
            f"L {cx - rx},{cy} Z"
        )

    aspect = _paper_aspect(fig)

    if simple:
        y0, y1 = 1.01, 1.08  # closer to plot (slightly up)
    else:
        y0, y1 = 1.03, 1.25
    y_center = 0.5 * (y0 + y1)

    pad_x = 0.01

    box_w = 0.28
    base_gap = 0.02
    group_gap = 0.22
    ar_group_gap = 0.09
    legend_shift_x = -0.1

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    ann = list(fig.layout.annotations) if fig.layout.annotations else []

    x_cursor = pad_x + legend_shift_x

    # Simple: center legend items w.r.t. the *entire figure* (including margins).
    # Plotly "paper" coords are relative to the plot area, so we convert
    # figure-center (px) into paper units, and estimate legend width in paper
    # units.
    if simple and s_present:
        inner_pad = 0.018
        icon_len = 0.05
        icon_gap = 0.015
        # horizontal gap between the two legend items (keep small)
        side_gap = 0.25

        # Labels in simple mode (must match what you actually draw below)
        block_label = "Block Diffusion"
        soft_label = "Set Diffusion"

        # Convert estimated text widths (px) -> paper units
        m = fig.layout.margin
        fig_w_px = float(fig.layout.width or 400)
        plot_w_px = fig_w_px - float(m.l + m.r)
        paper_per_px = (1.0 / plot_w_px) if plot_w_px > 0 else 0.0

        # Heuristic: estimate rendered text width.
        # 0.58em was too large for typical Plotly fonts and causes over-spacing;
        # use ~0.33em to better match actual rendered widths here.
        def _text_w_paper(text: str, font_size: float) -> float:
            est_px = 0.33 * font_size * len(text)
            return est_px * paper_per_px

        block_text_w = _text_w_paper(block_label, legend_text_size)
        soft_text_w = _text_w_paper(soft_label, legend_text_size)

        # Total legend width for the simple layout (single row, side-by-side items)
        total_width = (
            inner_pad
            + icon_len
            + icon_gap
            + block_text_w
            + side_gap
            + icon_len
            + icon_gap
            + soft_text_w
        )

        # Figure pixel center -> paper coordinate (paper x=0 at left edge of plot area)
        fig_center_paper = (
            ((fig_w_px / 2.0) - float(m.l)) / plot_w_px if plot_w_px > 0 else 0.5
        )
        x_cursor = fig_center_paper - 0.5 * total_width

        # Slight left adjustment so legend appears visually centered in the figure
        x_cursor -= 0.15

    if has_ar:
        ar_w = 0.04

        x_star = x_cursor + 0.020
        shapes.append(
            dict(
                type="path",
                xref="paper",
                yref="paper",
                path=_star_path(x_star, y_center, 0.016, 0.007, aspect),
                fillcolor=AR_COLOR,
                line=dict(color=AR_COLOR, width=1),
                layer="above",
            )
        )
        ann.append(
            dict(
                x=x_star + 0.028,
                y=y_center,
                xref="paper",
                yref="paper",
                text="AR",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=legend_font_size, color="#111", family=FONT_FAMILY),
            )
        )

        x_cursor += ar_w

        if len(s_present) > 0:
            x_cursor += ar_group_gap
        else:
            x_cursor += base_gap

    for i, s in enumerate(s_present):
        x0 = x_cursor
        x1 = x0 + box_w

        body_top = y1 - 0.03
        body_bot = y0 + 0.03
        y_mid = 0.5 * (body_top + body_bot)
        line_gap = 0.07
        y_row1 = y_mid + 0.5 * line_gap
        y_row2 = y_mid - 0.5 * line_gap

        inner_pad = 0.018
        icon_len = 0.05
        icon_gap = 0.015

        # Simple: both items on same row (y_center), side-by-side
        if simple:
            y_row1 = y_row2 = y_center
            x_icon0_1 = x0 + inner_pad
            x_icon1_1 = x_icon0_1 + icon_len
            x_text_1 = x_icon1_1 + icon_gap
        else:
            x_icon0_1 = x0 + inner_pad
            x_icon1_1 = x_icon0_1 + icon_len
            x_text_1 = x_icon1_1 + icon_gap

        bc = block_color_by_s.get(s, "#666")
        sc = soft_color_by_s.get(s, bc)

        shapes.append(
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=x_icon0_1,
                y0=y_row1,
                x1=x_icon1_1,
                y1=y_row1,
                line=dict(color=bc, width=3),
                layer="above",
            )
        )
        shapes.append(
            dict(
                type="path",
                xref="paper",
                yref="paper",
                path=_diamond_path(x_icon1_1, y_row1, 0.0135, 0.0135, aspect),
                line=dict(color=bc, width=2),
                fillcolor="white",
                layer="above",
            )
        )
        block_label = "Block Diffusion" if simple else f"Block Diffusion (S={s})"
        ann.append(
            dict(
                x=x_text_1,
                y=y_row1,
                xref="paper",
                yref="paper",
                text=block_label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=legend_text_size, color="#111", family=FONT_FAMILY),
            )
        )

        if simple:
            # In simple mode, put the second item right after the first label.
            x_icon0_2 = x_text_1 + block_text_w + side_gap
            x_icon1_2 = x_icon0_2 + icon_len
            x_text_2 = x_icon1_2 + icon_gap
            y_row2 = y_center
        else:
            x_icon0_2 = x0 + inner_pad
            x_icon1_2 = x_icon0_2 + icon_len
            x_text_2 = x_icon1_2 + icon_gap

        shapes.append(
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=x_icon0_2,
                y0=y_row2,
                x1=x_icon1_2,
                y1=y_row2,
                line=dict(color=sc, width=3),
                layer="above",
            )
        )
        r = 0.009
        ry = r / max(aspect, 1e-9)
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x_icon1_2 - r,
                x1=x_icon1_2 + r,
                y0=y_row2 - ry,
                y1=y_row2 + ry,
                fillcolor=sc,
                line=dict(color=sc, width=1),
                layer="above",
            )
        )
        soft_label = "Set Diffusion" if simple else f"Ours: Set Diffusion (S ≤ {s * 2})"
        ann.append(
            dict(
                x=x_text_2,
                y=y_row2,
                xref="paper",
                yref="paper",
                text=soft_label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=legend_text_size, color="#111", family=FONT_FAMILY),
            )
        )

        x_cursor = x1 if not simple else x_text_2 + soft_text_w

        if i < len(s_present) - 1:
            x_cursor += group_gap
        else:
            x_cursor += base_gap

    fig.update_layout(shapes=shapes, annotations=ann)


groups = [
    {
        "name": "Block Diffusion (S=4)",
        "x": np.array([1.0, 1.55, 1.78, 1.93, 2.05, 2.15, 2.25, 2.36, 2.47, 2.59]),
        "x_std": np.array([0.0, 0.19, 0.23, 0.25, 0.26, 0.28, 0.28, 0.29, 0.30, 0.31]),
        "speed_x": np.array(
            [40.51, 55.36, 61.10, 64.61, 67.06, 69.27, 71.68, 73.50, 75.33, 77.8]
        ),
        "speed_x_std": np.array(
            [0.21, 4.53, 4.87, 5.02, 5.31, 5.30, 5.51, 4.98, 4.85, 4.84]
        ),
        "y": np.array(
            [63.53, 63.61, 63.31, 61.56, 60.42, 60.05, 58.91, 57.62, 54.97, 52.62]
        ),
    },
    {
        "name": "Set Diffusion (match w/ S=4)",
        "x": np.array([1.0, 1.54, 1.78, 1.95, 2.08, 2.20, 2.33, 2.46, 2.60, 2.75]),
        "x_std": np.array([0.0, 0.18, 0.22, 0.24, 0.25, 0.27, 0.28, 0.28, 0.29, 0.30]),
        "conf": np.array([1e6, 0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]),
        "speed_x": np.array(
            [45.25, 69.79, 80.82, 87.47, 92.62, 96.81, 102.86, 107.24, 116.60, 122.25]
        ),
        "speed_x_std": np.array(
            [0.17, 7.73, 9.48, 9.93, 10.17, 10.12, 10.65, 11.43, 10.33, 11.26]
        ),
        "y": np.array(
            [66.41, 67.85, 67.10, 65.73, 64.37, 63.68, 61.56, 58.23, 58.23, 54.81]
        ),
    },
    {
        "name": "Block Diffusion (S=16)",
        "x": np.array([1.0, 1.83, 2.28, 2.61, 2.89, 3.18, 3.43, 3.71]),
        "x_std": np.array([0.0, 0.32, 0.43, 0.50, 0.58, 0.65, 0.68, 0.75]),
        "speed_x": np.array([42.3, 72.0, 87.01, 98.33, 108.82, 115.83, 122.54, 134.48]),
        "speed_x_std": np.array([0.21, 9.64, 12.78, 14.50, 16.49, 17.70, 19.38, 21.75]),
        "y": np.array([50.49, 50.27, 48.82, 46.78, 45.03, 43.67, 42.23, 39.27]),
    },
    {
        "name": "Set Diffusion (match w/ S=16)",
        "x": np.array([1.0, 1.69, 2.04, 2.30, 2.54, 2.78, 3.04, 3.32, 3.67, 4.06]),
        "x_std": np.array([0.0, 0.22, 0.31, 0.37, 0.41, 0.48, 0.53, 0.62, 0.73, 0.81]),
        "conf": np.array([1e6, 0.99, 0.95, 0.90, 0.85, 0.8, 0.75, 0.7, 0.65, 0.60]),
        "speed_x": np.array(
            [41.51, 69.74, 82.77, 92.93, 101.39, 111.24, 121.78, 132.43, 147.4, 160.68]
        ),
        "speed_x_std": np.array(
            [0.37, 8.52, 11.45, 12.54, 14.77, 17.41, 19.53, 20.39, 29.66, 32.25]
        ),
        "y": np.array(
            [61.94, 60.12, 60.73, 59.59, 57.9, 56.56, 54.13, 49.20, 44.58, 41.62]
        ),
    },
]

for g in groups:
    assert len(g[xaxis_key]) == len(g["y"]), (
        f"len of {g['name']} is {len(g[xaxis_key])} != {len(g['y'])}"
    )

groups = [ar_dict] + groups

# Simple version: only chosen block (Block Diffusion S + Set Diffusion match S),
# no AR, thinner plot
SIMPLE_BLOCK_COLOR = "#d62728"  # red
SIMPLE_SOFT_COLOR = "#2ca02c"  # green
if simple_version:
    groups = [
        g
        for g in groups
        if (
            _extract_s(g["name"]) == simple_s
            and (_is_block_diff(g["name"]) or _is_soft_block_diff(g["name"]))
        )
    ]
    # Remove first value (lowest speed) from each group
    for g in groups:
        n = len(g[xaxis_key])
        if n > 1:
            for key in list(g.keys()):
                if isinstance(g[key], np.ndarray) and len(g[key]) == n:
                    g[key] = g[key][1:]


def _compute_extents(groups_: list[dict]) -> tuple[float, float, float, float]:
    xs = np.concatenate([g[xaxis_key] for g in groups_])
    ys = np.concatenate([g["y"] for g in groups_])
    xstd = np.concatenate(
        [g.get(xaxis_std_key, np.zeros_like(g[xaxis_key])) for g in groups_]
    )
    x_lo = float(np.min(xs - xstd))
    x_hi = float(np.max(xs + xstd))
    y_lo = float(np.min(ys))
    y_hi = float(np.max(ys))
    return x_lo, x_hi, y_lo, y_hi


def _gray_hex(v: int) -> str:
    v = int(np.clip(v, 0, 255))
    return f"#{v:02x}{v:02x}{v:02x}"


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def _axis_gradient_shapes(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n_segments: int = 64,
    start_gray: int = 170,
    end_gray: int = 0,
    width: int = 3,
) -> list[dict]:
    shapes: list[dict] = []

    dx = x1 - x0
    dy = y1 - y0
    if dx <= 0 or dy <= 0:
        return shapes

    # X axis: left (gray) -> right (black) at y=y0
    for i in range(n_segments):
        t0 = i / n_segments
        t1 = (i + 1) / n_segments
        tm = 0.5 * (t0 + t1)
        v = int(round(_lerp(start_gray, end_gray, tm)))
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=_lerp(x0, x1, t0),
                y0=y0,
                x1=_lerp(x0, x1, t1),
                y1=y0,
                line=dict(color=_gray_hex(v), width=width),
                layer="above",
            )
        )

    # Y axis: bottom (gray) -> top (black) at x=x0
    for i in range(n_segments):
        t0 = i / n_segments
        t1 = (i + 1) / n_segments
        tm = 0.5 * (t0 + t1)
        v = int(round(_lerp(start_gray, end_gray, tm)))
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x0,
                y0=_lerp(y0, y1, t0),
                x1=x0,
                y1=_lerp(y0, y1, t1),
                line=dict(color=_gray_hex(v), width=width),
                layer="above",
            )
        )

    return shapes


x_lo, x_hi, y_lo, y_hi = _compute_extents(groups)
dx = max(1e-9, x_hi - x_lo)
dy = max(1e-9, y_hi - y_lo)
pad_x = 0.08 * dx
pad_y = 0.08 * dy
x0_axis, x1_axis = x_lo - pad_x, x_hi + pad_x
y0_axis, y1_axis = y_lo - pad_y, y_hi + pad_y

if xaxis_max is not None:
    x1_axis = float(xaxis_max)
if xaxis_min is not None:
    x0_axis = float(xaxis_min)

block_s = [_extract_s(g["name"]) for g in groups if _is_block_diff(g["name"])]
block_s = [s for s in block_s if s is not None]
soft_s = [_extract_s(g["name"]) for g in groups if _is_soft_block_diff(g["name"])]
soft_s = [s for s in soft_s if s is not None]

if simple_version:
    color_by_s = {simple_s: SIMPLE_BLOCK_COLOR}
    block_color_by_s = {simple_s: SIMPLE_BLOCK_COLOR}
    soft_color_by_s = {simple_s: SIMPLE_SOFT_COLOR}
else:
    color_by_s = _unified_color_by_s(block_s + soft_s)
    block_color_by_s = color_by_s
    soft_color_by_s = color_by_s


fig = go.Figure()

for group in groups:
    is_single_point = len(group[xaxis_key]) == 1
    name = group["name"]
    legendgroup, display_name, type_rank = _legend_grouping(name, simple=simple_version)
    s = _extract_s(name)

    trace_color = None
    if name == "AR":
        trace_color = AR_COLOR
        symbol = "star"
    elif _is_block_diff(name) and (s is not None):
        trace_color = block_color_by_s.get(s, color_by_s.get(s))
        symbol = "diamond"
    elif _is_soft_block_diff(name) and (s is not None):
        trace_color = soft_color_by_s.get(s, color_by_s.get(s))
        symbol = "square"
    fig.add_trace(
        go.Scatter(
            x=group[xaxis_key],
            y=group["y"],
            mode="lines+markers" if not is_single_point else "markers",
            name=display_name,
            legendgroup=legendgroup,
            legendrank=(
                0 if legendgroup == "AR" else (1000 * int(s or 0) + 10 * type_rank)
            ),
            line=(
                dict(color=trace_color, width=2)
                if (not is_single_point and trace_color)
                else dict(width=2)
            ),
            marker=(
                dict(
                    symbol=symbol,
                    size=10 if is_single_point else 8,
                    color=("white" if _is_block_diff(name) else trace_color),
                    line=dict(
                        color=trace_color, width=(2 if _is_block_diff(name) else 1)
                    ),
                )
                if trace_color
                else dict(symbol=symbol, size=10 if is_single_point else 10)
            ),
            error_x=dict(
                type="data",
                array=group[xaxis_std_key],
                visible=show_xerr,
            ),
        )
    )
fig.update_layout(
    width=350 if simple_version else 500,
    height=400 if simple_version else 500,
    font=dict(family=FONT_FAMILY),
    xaxis_title=(
        "Parallelism factor (↑)" if xaxis_key != "speed_x" else "Speed (tokens/sec; ↑)"
    ),
    yaxis_title="GSM8K Test Accuracy (↑)",
    legend_title="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
    margin=dict(t=35, l=45, r=8, b=0),
)
_add_custom_grouped_legend(
    fig,
    groups_=groups,
    block_color_by_s=block_color_by_s,
    soft_color_by_s=soft_color_by_s,
    simple=simple_version,
)


def _arrowhead_path_paper_aspect(
    xh,
    yh,
    xt,
    yt,
    aspect_y_over_x,
    head_len=0.05,
    head_w=0.034,
):
    dx = xh - xt
    dy = (yh - yt) * aspect_y_over_x
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return ""

    ux, uy = dx / n, dy / n
    px, py = -uy, ux

    xb = xh - head_len * ux
    yb_s = yh * aspect_y_over_x - head_len * uy

    x1 = xb + 0.5 * head_w * px
    y1 = (yb_s + 0.5 * head_w * py) / aspect_y_over_x
    x2 = xb - 0.5 * head_w * px
    y2 = (yb_s - 0.5 * head_w * py) / aspect_y_over_x

    return f"M {xh},{yh} L {x1},{y1} L {x2},{y2} Z"


# --- Improved tradeoff arrow placement ---
# Default (non-simple) placement (paper coordinates)
x_tail, y_tail = 0.72, 0.68
x_head, y_head = 0.95, 0.91

# Simple version: place arrow + text between the two curves (data-driven placement)
if simple_version:
    # Identify the two curves in the simple plot
    block_g = next(g for g in groups if _is_block_diff(g["name"]))
    soft_g = next(g for g in groups if _is_soft_block_diff(g["name"]))

    xb = np.asarray(block_g[xaxis_key], dtype=float)
    yb = np.asarray(block_g["y"], dtype=float)
    xs = np.asarray(soft_g[xaxis_key], dtype=float)
    ys = np.asarray(soft_g["y"], dtype=float)

    # Choose an x in the overlapping x-range so we're between both curves
    x_overlap_lo = max(float(np.min(xb)), float(np.min(xs)))
    x_overlap_hi = min(float(np.max(xb)), float(np.max(xs)))
    x_mid = 0.5 * (x_overlap_lo + x_overlap_hi)

    # Interpolate y-values at x_mid on each curve (ensure sorted for interp)
    ib = np.argsort(xb)
    is_ = np.argsort(xs)
    yb_mid = float(np.interp(x_mid, xb[ib], yb[ib]))
    ys_mid = float(np.interp(x_mid, xs[is_], ys[is_]))

    # Define the vertical "gap" between curves at x_mid
    y_low, y_high = (yb_mid, ys_mid) if yb_mid <= ys_mid else (ys_mid, yb_mid)
    gap = max(1e-6, y_high - y_low)

    # Arrow endpoints in data coords: slightly right-shifted and closer to ~45 degrees
    x_span = max(1e-9, (x1_axis - x0_axis))
    x_shift = 0.2 * x_span  # move right
    x_tail_d = x_mid + x_shift - 0.10 * x_span
    x_head_d = x_mid + x_shift + 0.10 * x_span

    # Get curve y-values near endpoints to keep arrow inside the between-curves band
    def _interp_safe(
        xd: float, xarr: np.ndarray, yarr: np.ndarray, idx: np.ndarray, fallback: float
    ) -> float:
        x_sorted = xarr[idx]
        y_sorted = yarr[idx]
        if xd < float(x_sorted[0]) or xd > float(x_sorted[-1]):
            return fallback
        return float(np.interp(xd, x_sorted, y_sorted))

    # Compute the between-curves band at each endpoint and clamp into it
    def _band_at(xd: float) -> tuple[float, float]:
        yb_x = _interp_safe(xd, xb, yb, ib, yb_mid)
        ys_x = _interp_safe(xd, xs, ys, is_, ys_mid)
        lo, hi = (yb_x, ys_x) if yb_x <= ys_x else (ys_x, yb_x)
        return float(lo), float(hi)

    lo_t, hi_t = _band_at(x_tail_d)
    lo_h, hi_h = _band_at(x_head_d)

    # Target y level: lower than center of the gap (move arrow slightly down)
    y_center_d = 0.5 * (y_low + y_high) - 0.25 * gap

    # Slightly *reduce* rise so the arrow angles a bit more downward (more horizontal),
    # which better matches the perceived text angle after screen-space rotation.
    # (Exact matching depends on axis scales; this is a visual heuristic.)
    rise = 0.14 * gap
    y_tail_d = y_center_d - 0.5 * rise
    y_head_d = y_center_d + 0.5 * rise

    # Clamp with margin away from curves
    margin = 0.12 * gap
    y_tail_d = float(np.clip(y_tail_d, lo_t + margin, hi_t - margin))
    y_head_d = float(np.clip(y_head_d, lo_h + margin, hi_h - margin))

    # Convert data coords -> paper coords (for the existing arrow drawing code)
    def _to_paper_x(xd: float) -> float:
        return (xd - x0_axis) / max(1e-9, (x1_axis - x0_axis))

    def _to_paper_y(yd: float) -> float:
        return (yd - y0_axis) / max(1e-9, (y1_axis - y0_axis))

    x_tail, y_tail = _to_paper_x(x_tail_d), _to_paper_y(y_tail_d)
    x_head, y_head = _to_paper_x(x_head_d), _to_paper_y(y_head_d)

    # Force the arrow slightly downward in *paper space* (not affected by
    # data-space clamping).
    # Also keep it inside the plot area.
    arrow_down_paper = 0.025
    y_tail = float(np.clip(y_tail - arrow_down_paper, 0.02, 0.98))
    y_head = float(np.clip(y_head - arrow_down_paper, 0.02, 0.98))

    # Optional: re-center x a hair if needed (comment out if you don't want it)
    # x_tail = float(np.clip(x_tail, 0.02, 0.98))
    # x_head = float(np.clip(x_head, 0.02, 0.98))

arrow_color = "#6f6f6f"
aspect = _paper_aspect(fig)


dx_p = x_head - x_tail
dy_p = y_head - y_tail
dx_s = dx_p
dy_s = dy_p * aspect
n_s = math.hypot(dx_s, dy_s)
ux_s, uy_s = dx_s / n_s, dy_s / n_s
shaft_back = 0.018
x_shaft_end = x_head - shaft_back * ux_s
y_shaft_end = y_head - (shaft_back * uy_s) / aspect

xm = 0.5 * (x_tail + x_head)
ym = 0.5 * (y_tail + y_head)

shapes = list(fig.layout.shapes) if fig.layout.shapes else []

shapes.append(
    dict(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_tail,
        y0=y_tail,
        x1=x_shaft_end,
        y1=y_shaft_end,
        line=dict(color=arrow_color, width=3),
        layer="above",
    )
)

_head_len = 0.065 if simple_version else 0.05
_head_w = 0.048 if simple_version else 0.034
shapes.append(
    dict(
        type="path",
        xref="paper",
        yref="paper",
        path=_arrowhead_path_paper_aspect(
            x_head,
            y_head,
            x_tail,
            y_tail,
            aspect,
            head_len=_head_len,
            head_w=_head_w,
        ),
        line=dict(width=0),
        fillcolor=arrow_color,
        layer="above",
    )
)

fig.update_layout(shapes=shapes)


angle_deg = math.degrees(math.atan2(dy_s, dx_s))

px_s = -dy_s / n_s
py_s = dx_s / n_s

offset_up = (
    -0.065
)  # move text slightly higher above the arrow (simple version wants more clearance)
offset_left = -0.1  # move slightly toward the tail (left along the arrow)

x_text = xm + offset_up * px_s + offset_left * ux_s
y_text = ym + (offset_up * py_s + offset_left * uy_s) / aspect

# Simple: nudge text a touch more upward (stabilizes visually vs. multi-line label)
if simple_version:
    y_text += 0.04

fig.add_annotation(
    x=x_text,
    y=y_text,
    xref="paper",
    yref="paper",
    text="Improved<br>Tradeoff",
    showarrow=False,
    textangle=-angle_deg,
    xanchor="center",
    yanchor="bottom",
    font=dict(size=20 if simple_version else 16, color=arrow_color, family=FONT_FAMILY),
)

_tick_size = 20 if simple_version else 16
_title_size = 22 if simple_version else 18
if xaxis_key != "speed_x":
    fig.update_xaxes(
        ticksuffix="x",
    )
fig.update_xaxes(
    tickfont=dict(size=_tick_size, family=FONT_FAMILY),
    title_font=dict(size=_title_size, family=FONT_FAMILY),
    range=[x0_axis, x1_axis],
    showline=False,
    zeroline=False,
)
fig.update_yaxes(
    ticksuffix="%",
    tickfont=dict(size=_tick_size, family=FONT_FAMILY),
    title_font=dict(size=_title_size, family=FONT_FAMILY),
    title_standoff=14 if simple_version else 10,
    range=[y0_axis, y1_axis],
    showline=False,
    zeroline=False,
)

axis_shapes = _axis_gradient_shapes(x0=x0_axis, x1=x1_axis, y0=y0_axis, y1=y1_axis)
existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
fig.update_layout(shapes=existing_shapes + axis_shapes)

arrow_frac = 0.02
fig.add_annotation(
    x=x1_axis,
    y=y0_axis,
    xref="x",
    yref="y",
    ax=x1_axis - arrow_frac * (x1_axis - x0_axis),
    ay=y0_axis,
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.0,
    arrowwidth=2,
    arrowcolor="#000000",
    text="",
)
fig.add_annotation(
    x=x0_axis,
    y=y1_axis,
    xref="x",
    yref="y",
    ax=x0_axis,
    ay=y1_axis - arrow_frac * (y1_axis - y0_axis),
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.0,
    arrowwidth=2,
    arrowcolor="#000000",
    text="",
)

fig.write_image(filename)
if save_pdf:
    fig.write_image(pdf_filename)
    print(f"Plot saved as {filename} and {pdf_filename}")
else:
    print(f"Plot saved as {filename}")
