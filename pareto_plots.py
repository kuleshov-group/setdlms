import numpy as np
import plotly.graph_objects as go

avg_block_size = 16
filename = f"line_plot_block{avg_block_size}.png"

# ----- Initialize data -----
groups = [
    {
        "name": "Block diff (S=16)",
        "x": np.array([1.0, 2.0]),
        "x_std": np.array([0.0, 0.0]),
        "y": np.array([50.57, 46.02]),
    },
    {
        "name": "Soft block diff: linear (match w/ S=16)",
        "x": np.array([1.0, 2.0]),
        "x_std": np.array([0.0, 0.0]),
        "y": np.array([61.64, 57.54]),
    },
    {
        "name": "Soft block diff: power (match w/ S=16)",
        "x": np.array([1.0, 2.0]),
        "x_std": np.array([0.0, 0.0]),
        "y": np.array([63.91, 60.88]),
    },
    {
        "name": "AR",
        "x": np.array([1.0]),
        "x_std": np.array([0.0]),
        "y": np.array([79.38]),
    }
]
# ----- Initialize data -----
groups = [
    {
        "name": "Block diff (S=4)",
        "x": np.array([2.0]),
        "x_std": np.array([0.0]),
        "y": np.array([53.07]),
    },
    {
        "name": "AR",
        "x": np.array([1.0]),
        "x_std": np.array([0.0]),
        "y": np.array([79.38]),
    }
]



# ----- Create figure -----
fig = go.Figure()

for group in groups:
    is_single_point = len(group["x"]) == 1

    fig.add_trace(
        go.Scatter(
            x=group["x"],
            y=group["y"],
            mode="lines+markers" if not is_single_point else "markers",
            name=group["name"],
            marker=dict(
                symbol="star" if is_single_point else "circle",
                size=10 if is_single_point else 6,
            ),
            error_x=dict(
                type="data",
                array=group["x_std"],
                visible=True,
            ),
        )
    )
# ----- Layout -----
fig.update_layout(
    title=f"GSM8K Accuracy-Parallelism Tradeoff",
    xaxis_title="Parallelism factor (↑)",
    yaxis_title="0-shot pass@1 (↑)",
    legend_title="Model",
)
fig.update_xaxes(ticksuffix="x")
fig.update_yaxes(ticksuffix="%")
# ----- Save as PNG -----
fig.write_image(filename)

print(f"Plot saved as {filename}")