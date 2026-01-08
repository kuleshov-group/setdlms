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

    colorscale = pc.sequential.Viridis
    n = max(noise_schedule.length - 1, 1)

    active = (move_chance != 0) & (move_chance != 1)
    max_overlap = int(active.sum(axis=1).max())

    y_first = move_chance[:, 0]
    auc_first = float(np.trapz(y_first, x))
    expected_active = auc_first * noise_schedule.length

    for i in range(noise_schedule.length):
        u = i / n
        color = pc.sample_colorscale(colorscale, u)[0]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=move_chance[:, i],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"token {i + 1}",
                legendgroup=f"token_{i + 1}",
                showlegend=(row == 1 and col == 1),  # legend only once
            ),
            row=row,
            col=col,
        )

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


# ============================================================
# Mega plot 1: ONLY d = m, packed grid (no empties)
# ============================================================
# vals = list(range(1, L + 1))  # 1..8
vals = [2, 3, 6]

plot_fig1 = False
if plot_fig1:
    ncols1 = 2
    # nrows1 = ((len(vals) * 2) + ncols1 - 1) // ncols1  # ceil
    nrows1 = len(vals)

    subplot_titles = []
    # for v in vals:
    #     subplot_titles += [
    #         f"Block diffusion",
    #         f"Staggered noise",
    #     ]

    for i, v in enumerate(vals):
        # r = i // ncols1 + 1
        r = i + 1
        c = 1
        ns_block_diffusion = LinearNoise(
            block_size=v,
            length=L,
            plot_schedule=False,
        )
        ts = torch.linspace(0, 1, 100000).unsqueeze(-1).repeat(1, L)
        y = ns_block_diffusion.total_noise(ts)
        block_diffusion_expected_active = float(np.trapz(y[:, 0], ts[:, 0])) * L
        block_diffusion_max_overlap = ((y != 0) & (y != 1)).sum(-1).max().item() #- 1

        ns_staggered_noise = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=v,
            k=1.0,
            length=L,
            plot_schedule=False,
        )
        y = ns_staggered_noise.total_noise(ts)
        staggered_noise_expected_active = float(np.trapz(y[:, 0], ts[:, 0])) * L
        staggered_noise_max_overlap = ((y != 0) & (y != 1)).sum(-1).max().item() # - 1
        subplot_titles += [
            f"<b>Block diffusion</b> <br> Expected # unmasked = {block_diffusion_expected_active:.1f} <br> Max lookahead = {block_diffusion_max_overlap:.1f}",
            f"<b>Staggered noise</b> <br> Expected # unmasked = {staggered_noise_expected_active:.1f} <br> Max lookahead = {staggered_noise_max_overlap:.1f}",
        ]



    fig1 = make_subplots(
        rows=nrows1,
        cols=ncols1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    for i, v in enumerate(vals):
        # r = i // ncols1 + 1
        r = i + 1
        c = 1
        ns = LinearNoise(
            block_size=v,
            length=L,
            plot_schedule=False,
        )

        add_schedule_subplot(
            fig1,
            ns,
            r,
            c,
            # title_prefix=f"Matched expected # unmasked w/ block size {v}<br>",
            ncols=ncols1,
        )
        c = 2
        ns = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=v,
            k=1.0,
            length=L,
            plot_schedule=False,
        )
        add_schedule_subplot(
            fig1,
            ns,
            r,
            c,
            ncols=ncols1,
        )


    # for i, v in enumerate(vals):
    #     r = i // ncols1 + 1
    #     c = i % ncols1 + 1

    #     ns = EaseOutPowerNoise(
    #         block_size=L,
    #         desired_block_size=v,
    #         k=1.0,
    #         length=L,
    #         plot_schedule=False,
    #     )
    #     add_schedule_subplot(
    #         fig1,
    #         ns,
    #         r,
    #         c,
    #         title_prefix=f"Matched expected # unmasked w/ block size {v}<br>",
    #         ncols=ncols1,
    #     )

    fig1.update_layout(
        title="",                     # no global title
        template="plotly",
        plot_bgcolor="white",
        height=300 * nrows1,
        width=900,

        # 🔑 THIS IS THE IMPORTANT PART
        margin=dict(
            t=75,    # <-- reserve vertical space ABOVE subplots
            b=60,
            l=80,
            r=40,
        ),

        showlegend=True,
        legend=dict(
            # title="Token index",
            orientation="h",
            x=0.5,
            y=1.15,              # <-- lives INSIDE the top margin band
            xanchor="center",
            yanchor="top",
            font=dict(size=12),
        ),
    )
    # ---- Force x/y ticks + labels on EVERY subplot in fig1 ----
    for r in range(1, nrows1 + 1):
        for c in range(1, ncols1 + 1):
            fig1.update_xaxes(
                showticklabels=True,
                showline=True,
                linecolor="black",
                linewidth=2,
                layer="above traces",
                zeroline=False,
                row=r,
                col=c,
            )
            fig1.update_yaxes(
                showticklabels=True,
                showline=True,
                linecolor="black",
                linewidth=2,
                layer="above traces",
                zeroline=False,
                row=r,
                col=c,
            )

    row_title_pad = 0.08 # vertical padding above the row's subplot titles (paper coords)

    for r, v in enumerate(vals, start=1):
        i1 = (r - 1) * ncols1 + 0
        i2 = (r - 1) * ncols1 + 1
        y_titles = max(
            fig1.layout.annotations[i1].y,
            fig1.layout.annotations[i2].y,
        )

        y = y_titles + row_title_pad

        fig1.add_annotation(
            x=0.5,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>Matching w/ block size {v}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=18),
        )

    # fig1.update_layout(
    #     title="",
    #     title_x=0.5,
    #     template="plotly",
    #     height=260 * nrows1,
    #     width=1000,
    #     margin=dict(t=140, b=60, l=80, r=40),  # <-- add top margin for mega-titles
    # )

    # Axis titles (avoid repeating on every subplot)
    # ---- Put axis TITLES on EVERY subplot in fig1 ----
    for r in range(1, nrows1 + 1):
        for c in range(1, ncols1 + 1):
            fig1.update_xaxes(
                title_text="t",
                row=r,
                col=c,
            )
            fig1.update_yaxes(
                title_text="Mask prob.",
                row=r,
                col=c,
            )

    fig1.write_image("power_schedule_1.jpg")


# ============================================================
# Mega plot 2: fix d=3, and plot all m >= d (single row)
# ============================================================
d_fixed = 2

# d_num_blocks = L // d_fixed
# d_area = 1 / (2 * d_num_blocks)
# k = 1
# w = d_area * (k + 1) / k
# int_min = (2 * w - 1) - (w / (k + 1)) + (1-w)**(k+1)/((k+1)*w**k)

# m_vals = [m for m in range(1, L + 1) if m >= d_fixed]
# m_vals = [4, 5, 6]
# int_mins = [-1.0, 0.0, 0.15] #, 0.5]
# int_mins = [-1.0, 0.0, 0.15]
# int_mins = [-1.0, 0.1]
int_mins = [-1.0, 0.1]
ncols2 = len(int_mins)

subplot_titles = []

for j, int_min in enumerate(int_mins, start=1):
    ns = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=d_fixed,
        max_block_size=L,
        length=L,
        plot_schedule=False,
        int_min=int_min if int_min >= 0.0 else None,
        k=1.0 if int_min < 0.0 else None,
    )
    ts = torch.linspace(0, 1, 100000).unsqueeze(-1).repeat(1, L)
    y = ns.total_noise(ts)
    max_overlap = ((y != 0) & (y != 1)).sum(-1).max().item() - 1
    expected_active = float(np.trapz(y[:, 0], ts[:, 0])) * L
    if j == 2:
        subplot_titles += [f'<b>Staggered noise: power</b><br>Expected # unmasked = {expected_active:.1f}<br><span style="color:green;">Max lookahead = {max_overlap:d}</span>']
        # subplot_titles += [f'<b>Unmasking width: {ns.b:.1f}</b><br>Expected # unmasked = {expected_active:.1f}<br><span style="color:green;">Max lookahead = {max_overlap:d}</span>']
    else:
        # subplot_titles += [f'<b>Unmasking width: {ns.b:.1f}</b><br>Expected # unmasked = {expected_active:.1f}<br><span style="color:red;">Max lookahead = {max_overlap:d}</span>']
        subplot_titles += [f'<b>Staggered noise: linear</b><br>Expected # unmasked = {expected_active:.1f}<br><span style="color:red;">Max lookahead = {max_overlap:d}</span>']
fig2 = make_subplots(
    rows=1,
    cols=ncols2,
    subplot_titles=subplot_titles,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.08,
)

for j, int_min in enumerate(int_mins, start=1):
    ns = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=d_fixed,
        max_block_size=L,
        length=L,
        plot_schedule=False,
        int_min=int_min if int_min >= 0.0 else None,
        k=1.0 if int_min < 0.0 else None,
    )
    add_schedule_subplot(
        fig2,
        ns,
        1,
        j,
        ncols=ncols2,
    )

fig2.update_layout(
    title=f"<b>Staggered noise matched w/ block size {d_fixed}</b> (expected # unmasked = {expected_active:.1f})",
    title_x=0.5,
    title_y=1.0,
    template="plotly",
    plot_bgcolor="white",
    height=350,
    width=max(900, 360 * ncols2),

    # reserve space for title + legend
    margin=dict(t=120, b=60, l=80, r=40),

    showlegend=True,
    legend=dict(
        # title="<b>Token index</b>",
        orientation="h",
        x=0.5,
        y=1.6,          # below title_y=0.97, above subplot titles
        xanchor="center",
        yanchor="top",
        font=dict(size=12),
    ),
)
fig2.update_xaxes(title_text="t", showline=True, linecolor="black", linewidth=2)
fig2.update_yaxes(title_text="Mask prob.", showline=True, linecolor="black", linewidth=2, range=[0, 1], rangemode="tozero", constrain="domain")

fig2.write_image("power_schedule_2.jpg")

print("Wrote power_schedule_1.jpg and power_schedule_2.jpg")