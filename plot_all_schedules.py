from plot_power_schedule import expected_active
from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise
import numpy as np
import torch
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def add_schedule_subplot(fig, noise_schedule, row, col, title_prefix="", ncols=1):
    x, move_chance = schedule_curves(noise_schedule, n_points=1000)

    colorscale = pc.sequential.Bluered
    n = max(noise_schedule.length - 1, 1)

    active = (move_chance != 0) & (move_chance != 1)
    max_overlap = int(active.sum(axis=1).max())

    y_first = move_chance[:, 0]
    auc_first = float(np.trapz(y_first, x))
    expected_active = auc_first * noise_schedule.length

    for i in range(noise_schedule.length):
        u = i / n
        color = pc.sample_colorscale(colorscale, u)[0]

        try:

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=move_chance[:, i],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"{i + 1}",
                    legendgroup=f"token_{i + 1}",
                    showlegend=(row == 1 and col == 1),  # legend only once
                ),
                row=row,
                col=col,
            )
        except Exception as e:
            print(f"Error adding trace: {e}")
            import ipdb ; ipdb.set_trace()

    # IMPORTANT: ncols must equal the number of columns in make_subplots
    # try:
    #     fig.layout.annotations[(row - 1) * ncols + (col - 1)].text = (
    #         f"{title_prefix}"
    #         f"Max # unmasked={max_overlap}<br>"
    #         f"Expected # unmasked={expected_active:.2f}"
    #     )
    # except Exception as e:
    #     print(f"Error adding annotation: {e}")
    #     import ipdb ; ipdb.set_trace()

# desired_block_sizes = [1,2,3,6]
desired_block_sizes = [1, 2.3, 3, 4]
widths=[0.25, 0.6, 0.8, 1.0]
# widths = [1/desired_block_sizes[-1], None, None, 1.0]
# widths = [1/desired_block_sizes[-1], 1/2, 2/3, 1.0]
# ks = [None, 0.5, 0.5, None]
ks = [None] * len(desired_block_sizes)
ks[0] = 1
nrows2 = 2
ncols2 = 2

subplot_titles = []

for j, desired_block_size in enumerate(desired_block_sizes, start=1):
    ns = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=desired_block_size,
        max_block_size=L,
        length=L,
        plot_schedule=False,
        b=widths[j-1],
        k=ks[j-1],
    )
    ts = torch.linspace(0, 1, 100000).unsqueeze(-1).repeat(1, L)
    y = ns.total_noise(ts)
    max_overlap = ((y != 0) & (y != 1)).sum(-1).max().item()
    expected_active = max(np.round(ns.k / (ns.k + 1) * ns.b * L, 1), 1.0)
    
    # expected_active = max(float(np.trapz(y[:, 0], ts[:, 0])) * L, 1.0)
    if desired_block_size == 1:
        subplot_title = f'<b>AR</b><br>C̄ = {expected_active:.1f} token(s)'
    elif desired_block_size == L:
        subplot_title = f'<b>Diffusion</b><br>C̄ = {expected_active:.1f} token(s)'
    else:
        text_color = "green"
        subplot_title = f'<b><span style="color:{text_color};"><i>Set Diffusion</i></span></b><br>C̄ = {expected_active:.1f} token(s)'
    subplot_titles += [subplot_title]

    # main_title_name = f"Matched to BD3LM block size {desired_block_size}"
    # if desired_block_size == 1:
    #     main_title_name = "Matched to AR"
    # elif desired_block_size == L:
    #     main_title_name = "Matched  Diffusion"
    # subplot_titles += [f'<b>{main_title_name}</b><br>Avg inference budget = {expected_active:.1f} token(s)']
fig2 = make_subplots(
    rows=nrows2,
    cols=ncols2,
    subplot_titles=subplot_titles,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    vertical_spacing=0.23,
)
for ann in fig2.layout.annotations:
    ann.yshift = 15  # increase this value for more space
row_ind = 1
col_ind = 1
for r, desired_block_size in enumerate(desired_block_sizes, start=1):
    ns = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=desired_block_size,
        max_block_size=L,
        length=L,
        plot_schedule=False,
        k=ks[r-1],
        b=widths[r-1],
        int_min=0.1 if (widths[r-1] is None and ks[r-1] is None) else None,
    )
    add_schedule_subplot(
        fig2,
        ns,
        row_ind,
        col_ind,
        ncols=ncols2,
    )
    if r % nrows2 == 0:
        row_ind += 1
        col_ind = 1
    else:
        col_ind += 1

fig2.update_layout(
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
        y=1.3,          # keep padding, reduce whitespace above legend
        xanchor="center",
        yanchor="top",
        font=dict(size=16),
    ),
)
# yaxis_title = r"$1 - \boldsymbol{\alpha}_t^i$"
yaxis_title = r"$\text{Mask probability}\,\, 1 - \boldsymbol{\alpha}_t^i$"
fig2.update_xaxes(title_text=r"$t$", showline=True, linecolor="black", linewidth=2, title_font_size=16)
fig2.update_yaxes(
    showline=True,
    linecolor="black",
    linewidth=2,
    range=[0, 1],
    rangemode="tozero",
    constrain="domain",
)
fig2.add_annotation(
    text=yaxis_title,
    x=-0.06,          # move left/right if needed
    y=0.5,            # centered vertically
    xref="paper",
    yref="paper",
    showarrow=False,
    textangle=-90,
    xanchor="center",
    yanchor="middle",
    font=dict(size=16),
)
# Remove x-axis titles everywhere
fig2.update_xaxes(title_text=None)

# Add x-axis title only to bottom row
for c in range(1, ncols2 + 1):
    fig2.update_xaxes(
        title_text=r"$t$",
        row=nrows2,
        col=c,
        title_font_size=16,
        title_standoff=0,
        automargin=True,
    )

fig2.update_layout(
    autosize=False,
    margin=dict(
        l=40,   # just enough for y-label
        r=10,
        t=0,
        b=45,
    ),
)
# fname = "all_schedules_new.png"
# fig2.write_image(fname)
# write to pdf
fig2.write_image("all_schedules.pdf")


# print(f"Wrote {fname}")