from collections import Counter

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.noise_schedule.noise_schedules import EaseOutPowerNoise

def all_nonzero_patterns_sorted(L: int):
    pats = []
    for m in range(1, 2**L):
        pat = tuple((m >> i) & 1 for i in range(L))
        pats.append(pat)

    def sort_key(pat):
        masked = [i for i, v in enumerate(pat) if v == 1]
        s = len(masked)
        rightmost = max(masked) if masked else -1
        return (s, rightmost)

    return sorted(pats, key=sort_key)

BASE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 22
SUBPLOT_TITLE_FONT_SIZE = 18
XLABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16

def sample_permutation_order(noise, L: int, num_samples: int = 100):
    t = torch.rand(1, L)
    max_deviations = []
    to_permute = torch.ones(L).unsqueeze(0)
    for _ in tqdm(range(num_samples), desc="Sampling permutation order"):
        perm_indices = noise.sample_permutation_order(t, to_permute)
        max_deviation = (perm_indices - torch.arange(0, noise.block_size)[None, None, :]).abs()
        max_deviations.extend(max_deviation.flatten().tolist())
    max_deviations = torch.tensor(max_deviations)
    print("max deviation: ", max_deviations.max().item())
    print("mean deviation: ", max_deviations.float().mean().item())
    print("std deviation: ", max_deviations.float().std().item())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=max_deviations.numpy(), nbinsx=100))
    fig.update_layout(
        xaxis_title="Max deviation",
        yaxis_title="Frequency",
        title="Max deviation distribution"
    )
    fig.write_image(f"max_deviation_distribution_L{L}_block_size{noise.block_size}.png")
    print(f"max_deviation_distribution_L{L}_block_size{noise.block_size}.png saved")

def simulate_pattern_probs_across_t(
    noise,
    L: int,
    t_steps: int = 1000,
    masking_trials_per_t: int = 2000,
    t_min_index: int = 1,
    t_max_index: int | None = None,
    clamp_probs: bool = True,
    device: str | torch.device = "cpu",
):
    if t_max_index is None:
        t_max_index = t_steps

    t_grid = torch.linspace(0, 1, t_steps, device=device).unsqueeze(-1).repeat(1, L)

    counts = Counter()
    num_predicted = []
    total = 0

    for ti in range(t_min_index, t_max_index):
        t = t_grid[ti].unsqueeze(0)
        p = noise.total_noise(t)
        alpha_t_prime = noise.rate_noise(t)
        if clamp_probs:
            p = p.clamp(0, 1)

        mask_samples = torch.rand(masking_trials_per_t, L, device=device) < p
        alpha_mask = (alpha_t_prime != 0.0).reshape(1, L).expand(masking_trials_per_t, L)
        num_predicted.extend((mask_samples & alpha_mask).sum(dim=-1).tolist())

        patterns_np = mask_samples.to(torch.int8).cpu().numpy()
        for row in patterns_np:
            counts[tuple(row.tolist())] += 1
        total += int(patterns_np.shape[0])

    ordered = all_nonzero_patterns_sorted(L)
    probs_dict = {pat: c / total for pat, c in counts.items()}
    patterns_cond = np.array([list(p) for p in ordered], dtype=int)
    probs_cond = np.array([probs_dict.get(p, 0.0) for p in ordered], dtype=float)
    patterns_cond = patterns_cond[probs_cond > 0]
    if len(patterns_cond) == L:
        probs_cond.fill(1.0 / L)
    return patterns_cond, probs_cond, num_predicted

def plot_pattern_probabilities(
    fig,
    patterns,
    probs,
    L: int,
    row: int,
    col: int,
    n_cols: int = 1,
    max_patterns: int | None = None,
    bar_width: float = 0.75,
    ymax: float | None = None,
):
    if max_patterns is not None:
        patterns = patterns[:max_patterns]
        probs = probs[:max_patterns]

    x = np.arange(len(patterns))
    probs = np.asarray(probs, dtype=float)
    if ymax is None:
        ymax = max(float(probs.max()) if len(probs) else 1e-8, 1e-8)

    ar_patterns = {
        (0, 0, 0, 1),
        (0, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    }
    bar_colors = [
        "orange" if tuple(pat) in ar_patterns else "rgb(31, 119, 180)"  # C0 equivalent
        for pat in patterns
    ]

    fig.add_trace(
        go.Bar(
            x=x,
            y=probs,
            marker_color=bar_colors,
            marker_line_width=0,
            width=bar_width,
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    subplot_idx = (row - 1) * n_cols + col
    xaxis_key = "xaxis" if subplot_idx == 1 else f"xaxis{subplot_idx}"
    xaxis = fig.layout[xaxis_key]
    x_dom0, x_dom1 = xaxis.domain

    yaxis_key = "yaxis" if subplot_idx == 1 else f"yaxis{subplot_idx}"
    d0_plot, d1_plot = fig.layout[yaxis_key].domain
    band = getattr(fig.layout, "_glyph_band", 0.18)
    y1_band = d0_plot
    y0_band = d0_plot - band * (d1_plot - d0_plot) / (1.0 - band)

    row_h = (y1_band - y0_band) / max(L, 1)
    square_y = row_h
    fig_w = float(fig.layout.width) if fig.layout.width is not None else 1.0
    fig_h = float(fig.layout.height) if fig.layout.height is not None else 1.0
    square_x = square_y * (fig_h / fig_w)
    n_patterns = len(patterns)
    for j, pat in enumerate(patterns):
        x_center = x_dom0 + ((j + 0.5) / n_patterns) * (x_dom1 - x_dom0)
        for t in range(L):
            y_center = y1_band - (t + 0.5) * row_h
            y0 = y_center - 0.5 * square_y
            y1 = y_center + 0.5 * square_y
            masked = bool(pat[t])
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=float(x_center - 0.5 * square_x),
                x1=float(x_center + 0.5 * square_x),
                y0=float(y0),
                y1=float(y1),
                line=dict(width=1),
                fillcolor="rgb(51, 51, 51)" if masked else "white",
                layer="above",
            )

    n_patterns = len(patterns)
    fig.update_xaxes(
        type="linear",
        range=[-0.5, n_patterns - 0.5],
        autorange=False,
        tickmode="linear",
        tick0=0,
        dtick=1,
        showticklabels=False,
        tickfont=dict(size=TICK_FONT_SIZE),
        showgrid=False,
        zeroline=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(range=[0, ymax * 1.15], row=row, col=col)


def plot_pattern_probabilities_multi_schedule(
    noise_schedules,
    L: int,
    t_steps: int = 1000,
    masking_trials_per_t: int = 2000,
    max_patterns: int | None = None,
    figsize=(11, 5),
    savepath: str | None = None,
    device: str | torch.device = "cpu",
):
    horizontal_spacing = 0.1
    nrows = 2
    ncols = 2

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.16,
        subplot_titles=[""] * (nrows * ncols),
    )

    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        font=dict(size=BASE_FONT_SIZE),
    )

    fig.layout._glyph_band = min(0.22, 0.045 * L + 0.04)

    band = fig.layout._glyph_band
    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            idx = (r - 1) * ncols + c
            ykey = "yaxis" if idx == 1 else f"yaxis{idx}"
            d0, d1 = fig.layout[ykey].domain
            new_d0 = d0 + band * (d1 - d0)
            fig.update_yaxes(domain=[new_d0, d1], row=r, col=c)

    fig.update_layout(
        showlegend=False,
        margin=dict(b=120, t=170),
        plot_bgcolor="white",
    )

    m = fig.layout.margin
    ml = m.l or 0
    mr = m.r or 0
    header_y = 1.16

    LEGEND_LABEL_FS = 18
    LEGEND_SQUARE_FS = 21
    LEGEND_AR_FS = 18
    d_legend = 0.0125

    plot_w = float(fig.layout.width) - (ml + mr)

    def _text_w_paper(txt: str, fs: float) -> float:
        return (0.60 * fs * len(txt)) / plot_w

    def _square_w_paper(fs: float) -> float:
        return (0.60 * fs) / plot_w

    masked_label = "Masked"
    clean_label = "Clean"

    sq_w = _square_w_paper(LEGEND_SQUARE_FS)
    m_tw = _text_w_paper(masked_label, LEGEND_LABEL_FS)
    c_tw = _text_w_paper(clean_label, LEGEND_LABEL_FS)

    item_gap = 0.08

    masked_item_w = sq_w + d_legend + m_tw
    clean_item_w = sq_w + d_legend + c_tw
    group_w = masked_item_w + item_gap + clean_item_w

    legend_center_x = 0.5
    group_left = legend_center_x - group_w / 2.0

    mask_x = group_left + sq_w / 2.0
    clean_x = group_left + masked_item_w + item_gap + sq_w / 2.0

    fig.add_annotation(
        xref="paper", yref="paper",
        x=mask_x, y=header_y,
        text="■",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=LEGEND_SQUARE_FS, color="rgb(51,51,51)"),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=mask_x + d_legend, y=header_y,
        text=masked_label,
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=LEGEND_LABEL_FS),
    )

    dx = 0.0
    dy = 0.0

    fig.add_annotation(
        xref="paper", yref="paper",
        x=clean_x, y=header_y,
        text="■",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=LEGEND_SQUARE_FS + 2, color="black"),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=clean_x + dx, y=header_y + dy,
        text="■",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=LEGEND_SQUARE_FS - 4, color="white"),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=clean_x + d_legend, y=header_y,
        text=clean_label,
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=LEGEND_LABEL_FS),
    )

    x_ar = -0.01
    fig.add_annotation(
        xref="paper", yref="paper",
        x=x_ar, y=header_y,
        text="■",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=LEGEND_SQUARE_FS, color="orange"),
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=x_ar + d_legend, y=header_y,
        text="AR masking",
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=LEGEND_AR_FS),
    )

    titles = []
    row_idx, col_idx = 1, 1
    all_patterns, all_probs = [], []
    for ns in noise_schedules:
        patterns, probs, num_predicted = simulate_pattern_probs_across_t(
            noise=ns,
            L=L,
            t_steps=t_steps,
            masking_trials_per_t=masking_trials_per_t,
            device=device,
        )
        all_patterns.append(patterns)
        all_probs.append(probs)
        expected_inference_prediction_budget = ns.compute_inf_budget()
        print("empirical inference prediction budget: ", np.mean(num_predicted), "+/-", np.std(num_predicted))
        print("expected inference prediction budget: ", expected_inference_prediction_budget)

    global_ymax = max(float(p.max()) if len(p) else 1e-8 for p in all_probs)

    for patterns, probs, ns in zip(all_patterns, all_probs, noise_schedules):
        expected_active = max(np.round(ns.k / (ns.k + 1) * ns.b * L, 1), 1.0)
        if ns.b <= 1 / L:
            title = f'<b>AR</b><br>C̄ = {expected_active:.1f} token(s)'
        elif ns.b == 1.0:
            title = f'<b>Masked Diffusion</b><br>C̄ = {expected_active:.1f} token(s)'
        else:
            text_color = "green"
            title = f'<b><span style="color:{text_color};"><i>Set Masked Diffusion</i></span></b><br>C̄ = {expected_active:.1f} token(s)'
        titles.append(title)
        print(row_idx, col_idx)
        plot_pattern_probabilities(
            fig=fig,
            patterns=patterns,
            probs=probs,
            L=L,
            row=row_idx,
            col=col_idx,
            n_cols=ncols,
            max_patterns=max_patterns,
            ymax=global_ymax,
        )

        col_idx += 1
        if col_idx > ncols:
            col_idx = 1
            row_idx += 1

    for i, title in enumerate(titles, start=1):
        xaxis_key = "xaxis" if i == 1 else f"xaxis{i}"
        yaxis_key = "yaxis" if i == 1 else f"yaxis{i}"
        x_dom0, x_dom1 = fig.layout[xaxis_key].domain
        x_center = 0.5 * (x_dom0 + x_dom1)
        y_top = fig.layout[yaxis_key].domain[1] - 0.

        fig.add_annotation(
            text=title,
            xref="paper",
            yref="paper",
            x=x_center,
            y=y_top,
            showarrow=False,
            font=dict(size=SUBPLOT_TITLE_FONT_SIZE),
            xanchor="center",
            yanchor="bottom",
        )

    fig.update_yaxes(
        title_text=r"$\text{Sampling prob. } q(\mathbf{z}_t | \mathbf{x})$",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text=r"$\text{Sampling prob. } q(\mathbf{z}_t | \mathbf{x})$",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        row=2,
        col=1,
    )

    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            fig.update_yaxes(showticklabels=True, tickfont=dict(size=TICK_FONT_SIZE), row=r, col=c)

    fig.add_annotation(
        text=r"$\text{Masked sequence } \mathbf{z}_t$",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0,
        yshift=-10,
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=XLABEL_FONT_SIZE),
    )
    fig.update_layout(
        margin=dict(l=35, r=15, t=70, b=40, pad=0),
    )

    if savepath is not None:
        if savepath.endswith('.html'):
            fig.write_html(savepath)
        else:
            try:
                fig.write_image(savepath, width=figsize[0] * 50, height=figsize[1] * 100, scale=2)
            except Exception as e:
                print(f"Warning: Could not save as image. Error: {e}")
                print("Falling back to HTML output. Install kaleido with: pip install kaleido")
                html_path = savepath.replace('.png', '.html').replace('.jpg', '.html')
                fig.write_html(html_path)
    print(f"Saved to {savepath}")

    return fig


if __name__ == "__main__":
    L = 4
    desired_block_sizes = [1, 2.3, 3, 4]
    widths = [0.25, 0.6, 0.8, 1.0]
    ks = [None] * len(desired_block_sizes)

    noise_schedules = []
    for desired_block_size, width, k in zip(desired_block_sizes, widths, ks):
        ns = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=desired_block_size,
            max_block_size=L,
            length=L,
            plot_schedule=False,
            b=width,
            int_min=0.1 if (width is None and k is None) else None,
            k=k,
        )
        noise_schedules.append(ns)
    plot_pattern_probabilities_multi_schedule(
        noise_schedules=noise_schedules,
        L=L,
        t_steps=4096,
        masking_trials_per_t=4096,
        max_patterns=None,
        savepath="masking_pattern_freqs.pdf",
    )