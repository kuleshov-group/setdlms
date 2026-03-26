import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.noise_schedule.noise_schedules import EaseOutPowerNoise

L = 4


def schedule_curves(noise_schedule, n_points=1000):
    num_blocks = noise_schedule.length // noise_schedule.block_size
    t = (
        torch.linspace(0, 1, n_points)
        .unsqueeze(-1)
        .repeat(1, noise_schedule.block_size)
        .repeat_interleave(num_blocks, dim=1)
    )
    move_chance = noise_schedule.total_noise(t).detach().cpu().numpy()
    x = np.linspace(0, 1, n_points)
    return x, move_chance


def add_schedule_subplot(fig, noise_schedule, row, col):
    x, move_chance = schedule_curves(noise_schedule, n_points=1000)

    colorscale = pc.sequential.Bluered
    n = max(noise_schedule.length - 1, 1)

    for i in range(noise_schedule.length):
        u = i / n
        color = pc.sample_colorscale(colorscale, u)[0]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=1 - move_chance[:, i],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{i + 1}",
                legendgroup=f"token_{i + 1}",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )


desired_block_sizes = [1, 2.3, 3, 4]
widths = [0.25, 0.6, 0.8, 1.0]
ks = [1, None, None, None]

nrows2 = 2
ncols2 = 2

schedules = []
subplot_titles = []
for desired_block_size, b, k in zip(desired_block_sizes, widths, ks):
    ns = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=desired_block_size,
        max_block_size=L,
        length=L,
        plot_schedule=False,
        b=b,
        k=k,
    )
    expected_active = max(np.round(ns.k / (ns.k + 1) * ns.b * L, 1), 1.0)

    if desired_block_size == 1:
        subplot_title = f"<b>AR</b><br>C̄ = {expected_active:.1f} token(s)"
    elif desired_block_size == L:
        subplot_title = f"<b>MDLM</b><br>C̄ = {expected_active:.1f} token(s)"
    else:
        subplot_title = (
            '<b><span style="color:green;"><i>SetDLM</i></span></b><br>'
            f"C̄ = {expected_active:.1f} token(s)"
        )

    schedules.append(ns)
    subplot_titles.append(subplot_title)

fig = make_subplots(
    rows=nrows2,
    cols=ncols2,
    subplot_titles=subplot_titles,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    vertical_spacing=0.3,
)
for ann in fig.layout.annotations:
    ann.yshift = 15

for idx, ns in enumerate(schedules):
    row_ind = (idx // ncols2) + 1
    col_ind = (idx % ncols2) + 1
    add_schedule_subplot(fig, ns, row_ind, col_ind)

fig.update_layout(
    title="",
    title_x=0.5,
    title_y=1.0,
    template="plotly",
    plot_bgcolor="white",
    height=450,
    width=500,
    # reserve space for title + legend
    # margin=dict(t=120, b=60, l=80, r=40),
    showlegend=True,
    legend=dict(
        title="<b>Token index</b>",
        orientation="h",
        x=0.5,
        y=1.3,  # keep padding, reduce whitespace above legend
        xanchor="center",
        yanchor="top",
        font=dict(size=16),
    ),
)
yaxis_title = r"$\text{Survival probability}\,\, \boldsymbol{\alpha}_t^i$"
fig.update_xaxes(showline=True, linecolor="black", linewidth=2, title_font_size=16)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    linewidth=2,
    range=[0, 1],
    rangemode="tozero",
    constrain="domain",
)
fig.add_annotation(
    text=yaxis_title,
    x=-0.06,  # move left/right if needed
    y=0.5,  # centered vertically
    xref="paper",
    yref="paper",
    showarrow=False,
    textangle=-90,
    xanchor="center",
    yanchor="middle",
    font=dict(size=16),
)
fig.update_xaxes(title_text=None)

for r in range(1, nrows2 + 1):
    for c in range(1, ncols2 + 1):
        fig.update_xaxes(
            title_text=r"$t$",
            showticklabels=True,
            row=r,
            col=c,
            title_font_size=16,
            title_standoff=0,
            automargin=True,
        )

fig.update_layout(
    autosize=False,
    margin=dict(
        l=40,  # just enough for y-label
        r=10,
        t=0,
        b=45,
    ),
)
fig.write_image("all_schedules.pdf")
