import numpy as np
import plotly.graph_objects as go
import re
from plotly.colors import sequential

avg_block_size = 4
filename = f"line_plot_block{avg_block_size}.png"
show_xerr = False

xaxis_key = "x"
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



groups = [
    {
        "name": "Block Diff. (S=4)",
        "x": np.array([1.0, 1.55, 1.64, 1.70, 1.74, 1.78, 1.93, 2.05, 2.15, 2.25, 2.36, 2.47, 2.59]),
        "x_std": np.array([0.0, 0.19, 0.21, 0.22, 0.22, 0.23, 0.25, 0.26, 0.28, 0.28, 0.29, 0.30, 0.31]),
        "speed_x": np.array([40.51, 55.36, 57.19, 59.83, 59.96, 61.10, 64.61, 67.06, 69.27, 71.68, 73.50, 75.33, 77.8]),
        "speed_x_std": np.array([0.21, 4.53, 4.59, 4.81, 4.72, 4.87, 5.02, 5.31, 5.30, 5.51, 4.98, 4.85, 4.84]),
        "conf": np.array([1e6, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]),
        "y": np.array([63.53, 63.61, 63.68, 63.38, 63.31, 63.31, 61.56, 60.42, 60.05, 58.91, 57.62, 54.97, 52.62]),
    },
    {
        "name": "Soft Block Diff. (match w/ S=4)",
        "x": np.array([1.0,2.64]),
        "x_std": np.array([0.0,0.28]),
        "speed_x": np.array([45.60, ...,111.14,110.88, 117.62]),
        "speed_x_std": np.array([0.23, ...11.30,10.88, 11.28]),
        "y": np.array([65.66,-1.0 53.68]),
    },
    # {
    #     "name": "Soft Block Diff. (match w/ S=4)",
    #     "x": np.array([1.71, 1.78, 1.95, 2.32, 2.44, 2.59, 2.71]),
    #     "x_std": np.array([0.24, 0.24, 0.27, 0.30, 0.30, 0.32, 0.32]),
    #     "speed_x": np.array([45.60, ..., 116.20]),
    #     "speed_x_std": np.array([0.23, ..., 10.66]),
    #     "y": np.array([66.41, 64.59, 64.82, 61.18, 58.53, 56.56, 52.77]),
    # },
    {
        "name": "Block Diff. (S=16)",
        "x": np.array([1.0, 1.83, 2.0, 2.11, 2.28, 2.61, 2.89, 3.18, 3.43, 3.71]),
        "x_std": np.array([0.0, 0.31, 0.36, 0.39, 0.43, 0.50, 0.58, 0.65, 0.68, 0.75]),
        "speed_x": np.array([42.25, 72.67, 82.41, 87.84, 98.87, 108.42, 116.75, 123.10, 132.87]),
        "speed_x_std": np.array([0.21, 9.66, 11.87, 12.78, 14.50, 16.49, 17.70, 19.38, 20.07]),
        "y": np.array([50.42, 50.19, 50.04, 49.43, 48.82, 46.78, 45.03, 43.59, 42.23, 39.27]),
    },
]

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



# ----- Color styling helpers -----
_S_RE = re.compile(r"S\s*=\s*(\d+)")


def _extract_s(name: str) -> int | None:
    m = _S_RE.search(name)
    return int(m.group(1)) if m else None


def _is_block_diff(name: str) -> bool:
    n = name.lower()
    return ("block diff" in n) and ("soft" not in n)


def _is_soft_block_diff(name: str) -> bool:
    return "soft block diff" in name.lower()


def _gray_shades_by_s(s_values: list[int]) -> dict[int, str]:
    # Light -> dark as S increases.
    if not s_values:
        return {}
    s_sorted = sorted(set(s_values))
    n = len(s_sorted)
    if n == 1:
        return {s_sorted[0]: "#7f7f7f"}

    light = 0.5  # near-white gray
    dark = 0.25   # dark gray
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
    palette = palette[2:] if len(palette) > 2 else palette
    if n == 1:
        return {s_sorted[0]: palette[len(palette) // 2]}
    idxs = np.linspace(0, len(palette) - 1, n).round().astype(int)
    return {s: palette[i] for s, i in zip(s_sorted, idxs)}


block_s = [_extract_s(g["name"]) for g in groups if _is_block_diff(g["name"])]
block_s = [s for s in block_s if s is not None]
soft_s = [_extract_s(g["name"]) for g in groups if _is_soft_block_diff(g["name"])]
soft_s = [s for s in soft_s if s is not None]

block_color_by_s = _gray_shades_by_s(block_s)
soft_color_by_s = _green_shades_by_s(soft_s)


# ----- Create figure -----
fig = go.Figure()

for group in groups:
    is_single_point = len(group[xaxis_key]) == 1
    name = group["name"]
    s = _extract_s(name)

    trace_color = None
    if _is_block_diff(name) and (s is not None):
        trace_color = block_color_by_s.get(s)
    elif _is_soft_block_diff(name) and (s is not None):
        trace_color = soft_color_by_s.get(s)

    fig.add_trace(
        go.Scatter(
            x=group[xaxis_key],
            y=group["y"],
            mode="lines+markers" if not is_single_point else "markers",
            name=name,
            line=dict(color=trace_color, width=2) if (not is_single_point and trace_color) else dict(width=2),
            marker=dict(
                symbol="star" if is_single_point else "circle",
                size=10 if is_single_point else 6,
                color=trace_color if trace_color else None,
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
    xaxis_title="Parallelism factor (↑)" if xaxis_key != "speed_x" else "Speed (tokens/sec; ↑)",
    yaxis_title="GSM8K 0-shot pass@1 (↑)",
    legend_title="",
    # White background (both plot area + surrounding paper)
    plot_bgcolor="white",
    paper_bgcolor="white",
    # Legend on top, horizontal
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=16),
    ),
)
if xaxis_key != "speed_x":
    fig.update_xaxes(
        ticksuffix="x",
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
fig.update_layout(shapes=axis_shapes)

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
# ----- Save as PNG -----
fig.write_image(filename)

print(f"Plot saved as {filename}")