import numpy as np
import plotly.graph_objects as go
import re
import math
from plotly.colors import sequential
from collections import defaultdict
import copy
# import plotly.io as pio
# pio.kaleido.scope.mathjax = None   # avoids some MathJax-related JS issues


avg_block_size = 4


xaxis_key = "speed_x"

filename = f"line_plot_block{avg_block_size}_{xaxis_key}.png"
show_xerr = False

# Optional vector export (requires kaleido, same as PNG export).
save_pdf = True
pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"

from typing import Optional, Tuple

# If set, force the maximum x value shown on the plot (in data units).
# Example: xaxis_max = 3.0
if xaxis_key == "x":
    xaxis_min = 0.8
    xaxis_max = 4.2
else:
    xaxis_min = 30
    xaxis_max = 170
AR_COLOR = "#636EFA"  # keep consistent with the trace


legend_font_size = 14

xaxis_std_key = xaxis_key + "_std"


# ----- Initialize data -----
ar_dict =  {
    "name": "AR",
    "x": np.array([1.0]),
    "x_std": np.array([0.0]),
    "speed_x": np.array([55.52]),
    "speed_x_std": np.array([0.25]),
    "y": np.array([80.21]),
}
# groups = [
#     {
#         "name": "Block diff (S=16)",
#         "x": np.array([2.61, 2.89]),
#         "x_std": np.array([0.50, 0.58])
#         "y": np.array([46.78, 45.03]),
#     },
# ]


def _rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    """
    SVG path for a rounded rectangle in paper coords.
    (Plotly doesn't have per-legend-group boxes, so we draw our own.)
    """
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    return (
        f"M {x0+r},{y0} "
        f"L {x1-r},{y0} "
        f"Q {x1},{y0} {x1},{y0+r} "
        f"L {x1},{y1-r} "
        f"Q {x1},{y1} {x1-r},{y1} "
        f"L {x0+r},{y1} "
        f"Q {x0},{y1} {x0},{y1-r} "
        f"L {x0},{y0+r} "
        f"Q {x0},{y0} {x0+r},{y0} Z"
    )
def _add_custom_grouped_legend(
    fig: go.Figure,
    *,
    groups_: list[dict],
    block_color_by_s: dict[int, str],
    soft_color_by_s: dict[int, str],
):
    s_present = sorted(
        {int(m.group(1)) for g in groups_ for m in [_S_RE.search(g["name"])] if m is not None}
    )
    has_ar = any(g.get("name") == "AR" for g in groups_)

    if not s_present and not has_ar:
        return

    # ---------------- helpers ----------------
    def _paper_aspect(fig: go.Figure) -> float:
        m = fig.layout.margin
        pw = float(fig.layout.width - (m.l + m.r))
        ph = float(fig.layout.height - (m.t + m.b))
        return ph / max(pw, 1e-9)

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
        return f"M {cx},{cy+ry_adj} L {cx+rx},{cy} L {cx},{cy-ry_adj} L {cx-rx},{cy} Z"

    aspect = _paper_aspect(fig)
    # -----------------------------------------

    # legend band
    y0, y1 = 1.02, 1.24
    y_center = 0.5 * (y0 + y1)

    pad_x = 0.01

    # ===== spacing knobs =====
    box_w = 0.28          # width of each S group box
    base_gap = 0.02       # small spacing at the very end
    group_gap = 0.22      # BIG spacing BETWEEN S-groups (e.g., S=4 <-> S=16)
    ar_group_gap = 0.09   # smaller spacing BETWEEN AR and first S-group
    legend_shift_x = -0.1
    # =========================

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    ann = list(fig.layout.annotations) if fig.layout.annotations else []

    x_cursor = pad_x + legend_shift_x

    # ---- AR (as its own group, but with smaller following gap) ----
    if has_ar:
        ar_w = 0.04

        x_star = x_cursor + 0.020
        shapes.append(dict(
            type="path", xref="paper", yref="paper",
            path=_star_path(x_star, y_center, 0.016, 0.007, aspect),
            fillcolor=AR_COLOR, line=dict(color=AR_COLOR, width=1),
            layer="above",
        ))
        ann.append(dict(
            x=x_star + 0.028, y=y_center,
            xref="paper", yref="paper",
            text="AR", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=legend_font_size, color="#111"),
        ))

        x_cursor += ar_w

        # AR -> first S group uses ar_group_gap (not group_gap)
        if len(s_present) > 0:
            x_cursor += ar_group_gap
        else:
            x_cursor += base_gap

    # ---- S groups ----
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

        x_icon0 = x0 + inner_pad
        x_icon1 = x_icon0 + icon_len
        x_text = x_icon1 + icon_gap

        bc = block_color_by_s.get(s, "#666")
        sc = soft_color_by_s.get(s, bc)

        # Block Diffusion
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=x_icon0, y0=y_row1, x1=x_icon1, y1=y_row1,
                           line=dict(color=bc, width=3), layer="above"))
        shapes.append(dict(type="path", xref="paper", yref="paper",
                           path=_diamond_path(x_icon1, y_row1, 0.0135, 0.0135, aspect),
                           line=dict(color=bc, width=2),
                           fillcolor="white", layer="above"))
        ann.append(dict(
            x=x_text, y=y_row1, xref="paper", yref="paper",
            text=f"Block Diffusion (S={s})",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=legend_font_size, color="#111"),
        ))

        # Set Diffusion
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=x_icon0, y0=y_row2, x1=x_icon1, y1=y_row2,
                           line=dict(color=sc, width=3), layer="above"))
        r = 0.009
        ry = r / max(aspect, 1e-9)
        shapes.append(dict(type="rect", xref="paper", yref="paper",
                           x0=x_icon1-r, x1=x_icon1+r,
                           y0=y_row2-ry, y1=y_row2+ry,
                           fillcolor=sc, line=dict(color=sc, width=1),
                           layer="above"))
        ann.append(dict(
            x=x_text, y=y_row2, xref="paper", yref="paper",
            text=f"Ours: Set Diffusion (S ≤ {s*2})",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=legend_font_size, color="#111"),
        ))

        x_cursor = x1

        # S-group -> next S-group uses BIG group_gap
        if i < len(s_present) - 1:
            x_cursor += group_gap
        else:
            x_cursor += base_gap

    fig.update_layout(shapes=shapes, annotations=ann)
def _legend_grouping(name: str) -> Tuple[Optional[str], str, Optional[str], int]:
    """
    Returns:
      legendgroup: group key (e.g., "S=4") or None
      display_name: trace label inside the group ("Block Diffusion" / "Set Diffusion" / "AR" / etc.)
      group_title: legend group title (shown once per group) or None
      type_rank: used for ordering within group (Block Diff before Soft Block Diff)
    """
    if name == "AR":
        return ("AR", "AR", None, 0)

    s = _extract_s(name)
    if s is None:
        # Fallback: keep as-is, no grouping
        return (None, name, None, 0)

    lg = f"Block size {s}"
    if _is_block_diff(name):
        # Put group title on the Block Diff trace so it appears once per S-group
        return (lg, f"Block Diffusion (S={int(s)})", None, 0)
    if _is_soft_block_diff(name):
        return (lg, f"Ours: Set Diffusion (S ≤ {int(s * 2)})", None, 1)

    # Unknown trace type but has S; group it anyway without a title
    return (lg, name, None, 2)



groups = [
    # {
    #     "name": "Block Diffusion (S=4)",
    #     "x": np.array([1.0, 1.55, 1.64, 1.70, 1.74, 1.78, 1.93, 2.05, 2.15, 2.25, 2.36, 2.47, 2.59]),
    #     "x_std": np.array([0.0, 0.19, 0.21, 0.22, 0.22, 0.23, 0.25, 0.26, 0.28, 0.28, 0.29, 0.30, 0.31]),
    #     "speed_x": np.array([40.51, 55.36, 57.19, 59.83, 59.96, 61.10, 64.61, 67.06, 69.27, 71.68, 73.50, 75.33, 77.8]),
    #     "speed_x_std": np.array([0.21, 4.53, 4.59, 4.81, 4.72, 4.87, 5.02, 5.31, 5.30, 5.51, 4.98, 4.85, 4.84]),
    #     "y": np.array([63.53, 63.61, 63.68, 63.38, 63.31, 63.31, 61.56, 60.42, 60.05, 58.91, 57.62, 54.97, 52.62]),
    # },
    {
        "name": "Block Diffusion (S=4)",
        "x": np.array([1.0, 1.55, 1.78, 1.93, 2.05, 2.15, 2.25, 2.36, 2.47, 2.59]),
        "x_std": np.array([0.0, 0.19, 0.23, 0.25, 0.26, 0.28, 0.28, 0.29, 0.30, 0.31]),
        "speed_x": np.array([40.51, 55.36, 61.10, 64.61, 67.06, 69.27, 71.68, 73.50, 75.33, 77.8]),
        "speed_x_std": np.array([0.21, 4.53, 4.87, 5.02, 5.31, 5.30, 5.51, 4.98, 4.85, 4.84]),
        "y": np.array([63.53, 63.61, 63.31, 61.56, 60.42, 60.05, 58.91, 57.62, 54.97, 52.62]),
    },
    {
        "name": "Set Diffusion (match w/ S=4)",
        "x": np.array([1.0, 1.54, 1.78, 1.95, 2.08, 2.20, 2.33, 2.46, 2.60, 2.75]),
        "x_std": np.array([0.0, 0.18, 0.22, 0.24, 0.25, 0.27, 0.28, 0.28, 0.29, 0.30]),
        "conf": np.array([1e6,0.99,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60]),
        "speed_x": np.array([45.25, 69.79, 80.82, 87.47, 92.62, 96.81, 102.86, 107.24, 116.60, 122.25]),
        "speed_x_std": np.array([0.17, 7.73, 9.48, 9.93, 10.17, 10.12, 10.65, 11.43, 10.33, 11.26]),
        "y": np.array([66.41, 67.85, 67.10, 65.73, 64.37, 63.68, 61.56, 58.23, 58.23, 54.81]),
    },      
    # {
    #     "name": "Set Diffusion (match w/ S=4)",
    #     "x": np.array([1.0,1.52,1.74, 1.91,2.03,2.15,2.26,2.40,2.54,2.64]),
    #     "x_std": np.array([0.0,0.16,0.21,0.23,0.25,0.26,0.27,0.28,0.29,0.28]),
    #     "speed_x": np.array([45.60, 67.35,77.11,84.58,90.21,95.04,99.87,105.37,111.14,117.62]),
    #     "speed_x_std": np.array([0.23,7.06,8.43,9.27,9.74,10.50,10.98,11.14,11.30, 11.28]),
    #     "y": np.array([65.66, 66.41, 64.59, 64.82, 64.44,63.38, 61.41,59.14,56.56,52.77]),
    # },
    # {
    #     "name": "Set Diffusion (match w/ S=4)",
    #     "x": np.array([1.0,2.40,2.57,2.77,2.99,3.24]),
    #     "x_std": np.array([0.0,0.37,0.42,0.45,0.48]),
    #     "conf": np.array([1e6,0.8,0.75,0.70,0.65,0.60,]),
    #     "speed_x": np.array([]),
    #     "speed_x_std": np.array([]),
    #     "y": np.array([64.44,54.74,51.33,46.47,37.53,33.13]),
    # },
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
        "x": np.array([1.0,1.69,2.04,2.30,2.54,2.78,3.04,3.32,3.67,4.06]),
        "x_std": np.array([0.0,0.22,0.31,0.37,0.41,0.48,0.53,0.62,0.73,0.81]),
        "conf": np.array([1e6,0.99,0.95,0.90,0.85,0.8,0.75,0.7,0.65,0.60]),
        "speed_x": np.array([41.51,69.74,82.77,92.93,101.39,111.24,121.78,132.43,147.4,160.68]),
        "speed_x_std": np.array([0.37,8.52,11.45,12.54,14.77,17.41,19.53,20.39,29.66,32.25]),
        "y": np.array([61.94,60.12,60.73,59.59,57.9,56.56,54.13,49.20,44.58,41.62]),
    },                                                                                            
    # {
    #     "name": "Set Diffusion (match w/ S=16)",
    #     "x": np.array([1.0, 1.69, 2.15, 2.54, 2.93])
    #     "x_std": np.array([0.0,0.23,-1.0,0.75,1.04]),
    #     "conf": np.array([1e6, 0.99, 0.95,0.90,0.85,0.80,0.75,0.70])
    #     "speed_x": np.array([40.84, 67.67, 87.73, 0.0, 0.0, 135.80,154.57, 178.79]),
    #     "speed_x_std": np.array([0.31, 7.62, 20.10, 0.0, 0.0, 53.84,69.0,83.22]),
    #     "y": np.array([60.8,60.58,61.11,59.74,57.24]),
    # },
]

# make sure everything has the same len
for g in groups:
    assert len(g[xaxis_key]) == len(g["y"]), f"len of {g['name']} is {len(g[xaxis_key])} != {len(g['y'])}"

groups = [ar_dict] + groups


# ----- Plot extents (for custom axis arrows/gradient baselines) -----
def _compute_extents(groups_: list[dict]) -> tuple[float, float, float, float]:
    xs = np.concatenate([g[xaxis_key] for g in groups_])
    ys = np.concatenate([g["y"] for g in groups_])
    xstd = np.concatenate([g.get(xaxis_std_key, np.zeros_like(g[xaxis_key])) for g in groups_])
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
    """
    Draw x- and y- axis baseline lines with a gray->black gradient in the + direction.
    We use many short line segments because Plotly axis lines can't be true gradients.
    """
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



# ----- Color styling helpers -----
_S_RE = re.compile(r"S\s*=\s*(\d+)")

# Fixed per-block-size colors (applies to both Block Diff and Soft Block Diff).
# Requested: S=4 -> orange, S=16 -> blue.
FIXED_COLOR_BY_S: dict[int, str] = {
    4: "#ff7f0e",   # Plotly orange
    16: "#1f77b4",  # Plotly blue
}


def _extract_s(name: str) -> int | None:
    m = _S_RE.search(name)
    return int(m.group(1)) if m else None


def _is_block_diff(name: str) -> bool:
    n = name.lower()
    return ("block diff" in n) and ("set" not in n)


def _is_soft_block_diff(name: str) -> bool:
    return "set" in name.lower()


def _gray_shades_by_s(s_values: list[int]) -> dict[int, str]:
    # Light -> dark as S increases.
    if not s_values:
        return {}
    s_sorted = sorted(set(s_values))
    n = len(s_sorted)
    if n == 1:
        return {s_sorted[0]: "#7f7f7f"}

    light = 0.6  # near-white gray
    dark = 0.1   # dark gray
    levels = light - (light - dark) * (np.arange(n) / (n - 1))
    colors = []
    for lvl in levels:
        v = int(round(255 * float(lvl)))
        colors.append(f"#{v:02x}{v:02x}{v:02x}")
    return dict(zip(s_sorted, colors))


def _green_shades_by_s(s_values: list[int]) -> dict[int, str]:
    # Use Plotly sequential greens; light -> dark as S increases.
    if not s_values:
        return {}
    s_sorted = sorted(set(s_values))
    n = len(s_sorted)
    palette = list(sequential.Greens)
    # Avoid the very lightest entries so points remain visible on white.
    palette = palette[3:-2] if len(palette) > 2 else palette
    if n == 1:
        return {s_sorted[0]: palette[len(palette) // 2]}
    idxs = np.linspace(0, len(palette) - 1, n).round().astype(int)
    return {s: palette[i] for s, i in zip(s_sorted, idxs)}

def _unified_color_by_s(s_values: list[int]) -> dict[int, str]:
    """
    Return a single color per block size S, used for both Block Diff and Set Diffusion
    Prefers `FIXED_COLOR_BY_S`; falls back to a palette for any other S.
    """
    if not s_values:
        return {}
    s_sorted = sorted(set(s_values))
    # Fallback palette (in case other S values appear).
    palette = [
        "#ff7f0e",  # orange
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    out: dict[int, str] = {}
    for i, s in enumerate(s_sorted):
        out[s] = FIXED_COLOR_BY_S.get(s, palette[i % len(palette)])
    return out


block_s = [_extract_s(g["name"]) for g in groups if _is_block_diff(g["name"])]
block_s = [s for s in block_s if s is not None]
soft_s = [_extract_s(g["name"]) for g in groups if _is_soft_block_diff(g["name"])]
soft_s = [s for s in soft_s if s is not None]

# Single color per S across both trace types (and legend icons).
color_by_s = _unified_color_by_s(block_s + soft_s)
block_color_by_s = color_by_s
soft_color_by_s = color_by_s


# ----- Create figure -----
fig = go.Figure()

for group in groups:
    is_single_point = len(group[xaxis_key]) == 1
    name = group["name"]
    legendgroup, display_name, group_title, type_rank = _legend_grouping(name)
    s = _extract_s(name)

    trace_color = None
    if name == "AR":
        trace_color = AR_COLOR
        symbol = "star"
    elif _is_block_diff(name) and (s is not None):
        trace_color = color_by_s.get(s)
        symbol = "diamond"
    elif _is_soft_block_diff(name) and (s is not None):
        trace_color = color_by_s.get(s)
        symbol = "square"
    fig.add_trace(
        go.Scatter(
            x=group[xaxis_key],
            y=group["y"],
            mode="lines+markers" if not is_single_point else "markers",
            name=display_name,
            legendgroup=legendgroup,
            # Title appears once per legendgroup (we set it only on Block Diff traces)
            legendgrouptitle_text=group_title,
            # Keep groups ordered by S, and within group: Block Diff then Soft Block Diff
            # (AR stays first because its legendrank is small)
            legendrank=(0 if legendgroup == "AR" else (1000 * int(s or 0) + 10 * type_rank)),

            line=dict(color=trace_color, width=2) if (not is_single_point and trace_color) else dict(width=2),
            marker=(
                dict(
                    symbol=symbol,
                    size=10 if is_single_point else 8,
                    # For Block Diff, use a white-filled square so the line doesn't show through.
                    color=("white" if _is_block_diff(name) else trace_color),
                    line=dict(color=trace_color, width=(2 if _is_block_diff(name) else 1)),
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
# ----- Layout -----
fig.update_layout(
    # title=f"GSM8K Accuracy-Parallelism Tradeoff",
    # Make figure taller + skinnier (in pixels) for PNG export.
    width=500,
    height=500,
    xaxis_title="Parallelism factor (↑)" if xaxis_key != "speed_x" else "Speed (tokens/sec; ↑)",
    yaxis_title="Accuracy (↑)",
    legend_title="",
    # White background (both plot area + surrounding paper)
    plot_bgcolor="white",
    paper_bgcolor="white",
    # Legend on top, horizontal
    showlegend=False,
    # Give enough room for the custom boxed legend band we draw in paper coords.
    margin=dict(t=90, l=55, r=20, b=55),
)
_add_custom_grouped_legend(fig, groups_=groups, block_color_by_s=block_color_by_s, soft_color_by_s=soft_color_by_s)

# ===== Diagonal "Improved Tradeoff" arrow (clean + aligned) =====
import math

def _paper_aspect(fig: go.Figure) -> float:
    """pixel aspect of plotting area (paper-y per paper-x)."""
    m = fig.layout.margin
    pw = float(fig.layout.width  - (m.l + m.r))
    ph = float(fig.layout.height - (m.t + m.b))
    return ph / max(pw, 1e-9)

def _arrowhead_path_paper_aspect(
    xh, yh, xt, yt, aspect_y_over_x,
    head_len=0.05,
    head_w=0.034,
):
    """
    Arrowhead triangle aligned with the visible shaft direction.
    """
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


# ---- placement ----
x_tail, y_tail = 0.72, 0.68
x_head, y_head = 0.95, 0.91

arrow_color = "#6f6f6f"
aspect = _paper_aspect(fig)


dx_p = x_head - x_tail
dy_p = y_head - y_tail
dx_s = dx_p
dy_s = dy_p * aspect
n_s = math.hypot(dx_s, dy_s)
ux_s, uy_s = dx_s / n_s, dy_s / n_s  # unit direction in screen-like coords
shaft_back = 0.018  # shorten the shaft near the head (paper-x units; tuned visually)
x_shaft_end = x_head - shaft_back * ux_s
y_shaft_end = y_head - (shaft_back * uy_s) / aspect

xm = 0.5 * (x_tail + x_head)
ym = 0.5 * (y_tail + y_head)

# ---- draw arrow shaft ----
shapes = list(fig.layout.shapes) if fig.layout.shapes else []

shapes.append(dict(
    type="line",
    xref="paper", yref="paper",
    x0=x_tail, y0=y_tail,
    x1=x_shaft_end, y1=y_shaft_end,
    line=dict(color=arrow_color, width=3),
    layer="above",
))

# ---- draw arrowhead ----
shapes.append(dict(
    type="path",
    xref="paper", yref="paper",
    path=_arrowhead_path_paper_aspect(
        x_head, y_head, x_tail, y_tail, aspect
    ),
    line=dict(width=0),
    fillcolor=arrow_color,
    layer="above",
))

fig.update_layout(shapes=shapes)


angle_deg = math.degrees(math.atan2(dy_s, dx_s))

px_s = -dy_s / n_s
py_s =  dx_s / n_s

offset_up = -0.04   # move above the arrow
offset_left = -0.1  # move slightly toward the tail (left along the arrow)

x_text = xm + offset_up * px_s + offset_left * ux_s
y_text = ym + (offset_up * py_s + offset_left * uy_s) / aspect

fig.add_annotation(
    x=x_text,
    y=y_text,
    xref="paper",
    yref="paper",
    text="Improved Tradeoff",
    showarrow=False,
    textangle=-angle_deg,  # EXACT match to arrow's rendered angle
    xanchor="center",
    yanchor="bottom",
    font=dict(size=16, color=arrow_color),
)

if xaxis_key != "speed_x":
    fig.update_xaxes(
        ticksuffix="x",
    )
fig.update_xaxes(
    tickfont=dict(size=16),
    title_font=dict(size=18),
    range=[x0_axis, x1_axis],
    showline=False,  # we draw custom gradient + arrow baselines instead
    zeroline=False,
)
fig.update_yaxes(
    ticksuffix="%",
    tickfont=dict(size=16),
    title_font=dict(size=18),
    range=[y0_axis, y1_axis],
    showline=False,  # we draw custom gradient + arrow baselines instead
    zeroline=False,
)

# ----- Custom axis baselines: gradient + arrowheads -----
axis_shapes = _axis_gradient_shapes(x0=x0_axis, x1=x1_axis, y0=y0_axis, y1=y1_axis)
existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
fig.update_layout(shapes=existing_shapes + axis_shapes)

# Arrowheads (draw only a short final segment so we don't override the gradient)
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

# ----- Save -----
fig.write_image(filename)
if save_pdf:
    fig.write_image(pdf_filename)
    print(f"Plot saved as {filename} and {pdf_filename}")
else:
    print(f"Plot saved as {filename}")