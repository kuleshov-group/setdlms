from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise, StaggeredNoise
from tqdm import tqdm
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sample_permutation_order(noise, L, num_samples=100):
    t = torch.rand(1, L)
    max_deviations = []
    to_permute = torch.ones(L).unsqueeze(0)
    for _ in tqdm(range(num_samples), desc="Sampling permutation order"):
        perm_indices = noise.sample_permutation_order(t, to_permute)
        max_deviation = (perm_indices - torch.arange(0, noise.block_size)[None, None, :]).abs()
        # max_deviations.append(max_deviation.max())
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

from collections import Counter
import torch
import re

def simulate_pattern_probs_across_t(
    noise,
    L: int,
    t_steps: int = 1000,
    masking_trials_per_t: int = 2000,
    t_min_index: int = 1,
    t_max_index: int | None = None,   # exclusive
    clamp_probs: bool = True,
    device: str | torch.device = "cpu",
):
    """
    Empirically estimate P(z) where z is a masking pattern in {0,1}^L,
    averaged uniformly over a t-grid, using Monte Carlo.

    Returns:
        patterns: list[list[int]]  (each length L)
        probs:    list[float]      (same length as patterns)
    """
    if t_max_index is None:
        t_max_index = t_steps

    t_grid = torch.linspace(0, 1, t_steps, device=device).unsqueeze(-1).repeat(1, L)

    counts = Counter()
    num_predicted = []
    total = 0

    for ti in range(t_min_index, t_max_index):
        t = t_grid[ti].unsqueeze(0)  # [1, L]
        # p = noise.total_noise(t).squeeze(0)  # [L]
        p, alpha_t_prime = noise.total_noise(t), noise.rate_noise(t)
        if clamp_probs:
            p = p.clamp(0, 1)
        masking_trials_per_t_kept = 0

        # sample masks at this t
        mask_samples = (torch.rand(masking_trials_per_t, L, device=device) < p)
        patterns_np = mask_samples.to(torch.int8).cpu().numpy()
        valid_to_predict = (alpha_t_prime != 0.0) & (patterns_np == 1)
        num_predicted.extend(valid_to_predict.sum(dim=-1).tolist())

        for row in patterns_np:
            # if row.sum() == 0:
            #     continue
            masking_trials_per_t_kept += 1
            counts[tuple(row.tolist())] += 1
        total += masking_trials_per_t_kept

    def sort_key(kv):
        pat = kv[0]
        masked = [i for i, v in enumerate(pat) if v == 1]
        s = len(masked)
        rightmost = max(masked) if masked else -1
        return (s, rightmost)

    # sort by structure ONLY (x-axis semantics), not by probability
    items = sorted(counts.items(), key=sort_key)

    patterns = [list(k) for k, _ in items]
    probs = [c / total for _, c in items]

    patterns = np.array(patterns)
    probs = np.array(probs)
    if len(patterns) == L:
        probs = np.ones(L) / L
        patterns = np.tril(np.ones((L,L)),0).astype(int)

    # identify the all-clean pattern
    is_all_clean = (patterns.sum(axis=1) == 0)

    p0 = probs[is_all_clean].sum()

    # keep only patterns with at least one masked token
    patterns_cond = patterns[~is_all_clean]
    probs_cond = probs[~is_all_clean] / (1.0 - p0)
    return patterns_cond, probs_cond, num_predicted

import numpy as np

def plot_pattern_probabilities(
    fig,
    patterns,
    probs,
    L: int,
    row: int,
    col: int,
    n_cols: int = 1,
    horizontal_spacing: float = 0.1,
    title: str | None = None,
    max_patterns: int | None = None,  # optionally truncate (e.g., top 16)
    bar_width: float = 0.75,
    square_size: float = 0.1,
    square_gap: float = 0.015,
    glyph_gap: float = 0.03,
):
    """
    Single bar chart:
      x: masking patterns (rendered as glyphs under bars)
      y: empirical probabilities
    """
    if max_patterns is not None:
        patterns = patterns[:max_patterns]
        probs = probs[:max_patterns]

    x = np.arange(len(patterns))
    probs = np.asarray(probs, dtype=float)

    # Highlight AR-ordering patterns in orange
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
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=x,
            y=probs,
            marker_color=bar_colors,
            marker_line_width=0,
            opacity=0.6,
            width=bar_width,
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    ymax = max(float(probs.max()) if len(probs) else 1e-8, 1e-8)
    
   # --- Glyphs: draw as layout shapes in paper coords (won't affect y-axis range)
    subplot_idx = (row - 1) * n_cols + col
    xaxis = fig.layout[f"xaxis{subplot_idx}" if subplot_idx > 1 else "xaxis"]
    x_dom0, x_dom1 = xaxis.domain

    band = getattr(fig.layout, "_glyph_band", 0.18)  # set once in the caller
    pad = 0.0
    y0_band = 0.0
    y1_band = band
    row_h = (y1_band - y0_band) / max(L, 1)

    square_y = row_h
    fig_w = float(fig.layout.width) if fig.layout.width is not None else 1.0
    fig_h = float(fig.layout.height) if fig.layout.height is not None else 1.0
    square_x = square_y * (fig_h / fig_w)  # compensate for paper-unit aspect
    n_patterns = len(patterns)
    for j, pat in enumerate(patterns):
        # map bar center to paper-x
        x_center = x_dom0 + (j + 0.5) / n_patterns * (x_dom1 - x_dom0)
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

    # Update subplot layout
    fig.update_xaxes(
        showticklabels=False,
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
    figsize=(18, 5),
    savepath: str | None = None,
    device: str | torch.device = "cpu",
):
    """
    One subplot per schedule. Each subplot is a single pattern-probability bar chart.
    """

    n = len(noise_schedules)
    horizontal_spacing = 0.1

    # Create empty subplot titles - we'll add colored annotations separately
    fig = make_subplots(
        rows=1,
        cols=n,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        subplot_titles=[] * n,  # Empty titles, we'll add colored ones as annotations
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
    )

    # reserve a bottom band for the glyph grid (shared across subplots)
    fig.layout._glyph_band = min(0.22, 0.045 * L + 0.04)
    for c in range(1, n + 1):
        fig.update_yaxes(domain=[fig.layout._glyph_band, 1.0], row=1, col=c)

    # --- Clean header layout (no Plotly legend)
    fig.update_layout(
        showlegend=False,
        margin=dict(b=120, t=210),  # room for: mega title + legend row + subplot titles
    )

    # --- Mega title (annotation so we can place it above the plot area)
    # fig.add_annotation(
    #     # text="Masking pattern probabilities across noise schedules",
    #     xref="paper", yref="paper",
    #     x=0.5, y=1.33,
    #     showarrow=False,
    #     xanchor="center", yanchor="top",
    #     font=dict(size=26, weight="bold"),
    # )

    # --- Header row y position (shared)
    header_y = 1.36

    # --- Custom legend centered: [■ Masked]  [□ Clean]  (with spacing)
    # Masked square - accounting for aspect ratio (fig is 18:5, so width in paper coords must be smaller)
    # Height in paper coords: 0.03, which is 0.03 * 400px = 12px
    # For square: width should also be 12px = 12/1440 = 0.00833 in paper coords
    square_width = 0.06 * (figsize[1] / figsize[0])  # Account for aspect ratio
    square_half_width = square_width / 2
    masked_center_x = 0.4475  # Center of masked square
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=masked_center_x - square_half_width, x1=masked_center_x + square_half_width,
        y0=header_y-0.015, y1=header_y+0.015,
        fillcolor="rgb(51, 51, 51)", line=dict(width=0),
        layer="above",
    )
    fig.add_annotation(
        text="Masked",
        xref="paper", yref="paper",
        x=0.47, y=header_y,
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=14),
    )

    # Clean square (outlined) - accounting for aspect ratio to make it square
    clean_center_x = 0.5875  # Center of clean square
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=clean_center_x - square_half_width, x1=clean_center_x + square_half_width,
        y0=header_y-0.015, y1=header_y+0.015,
        fillcolor="white", line=dict(color="rgb(128, 128, 128)", width=2),
        layer="above",
    )
    fig.add_annotation(
        text="Clean",
        xref="paper", yref="paper",
        x=0.605, y=header_y,
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=14),
    )

    # --- Autoregressive ordering key on the left (same baseline as legend)
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.02, x1=0.035, y0=header_y-0.015, y1=header_y+0.015,
        fillcolor="orange",
        line=dict(width=0),
        layer="above",
    )
    fig.add_annotation(
        text="Autoregressive masking",
        xref="paper", yref="paper",
        x=0.04, y=header_y,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=14),
    )

    titles = []  # Collect titles with pattern counts
    for col_idx, ns in enumerate(noise_schedules, start=1):
        patterns, probs, num_predicted = simulate_pattern_probs_across_t(
            noise=ns,
            L=L,
            t_steps=t_steps,
            masking_trials_per_t=masking_trials_per_t,
            device=device,
        )
        expected_inference_prediction_budget = ns.compute_inf_budget()
        print("empirical inference prediction budget: ", np.mean(num_predicted), "+/-", np.std(num_predicted))
        print("expected inference prediction budget: ", expected_inference_prediction_budget)

        num_masking_patterns = len(patterns)
        num_total_patterns = 2**L - 1
        percent_masking_patterns = num_masking_patterns / num_total_patterns * 100
        if col_idx == n:
            color = "green"
            title = f'Staggered noise: power<br><span style="color:{color};">Masking patterns possible: {int(percent_masking_patterns):d}%</span>'    
        else:
            color = "red"
            title = f'Staggered noise: linear<br><span style="color:{color};">Masking patterns possible: {int(percent_masking_patterns):d}%</span>'    

        # title = f'Unmasking width: {ns.b:.1f}<br><span style="color:{color};">Masking patterns possible: {int(percent_masking_patterns):d}%</span>'    
        titles.append(title)  # Store title for later annotation

        plot_pattern_probabilities(
            fig=fig,
            patterns=patterns,
            probs=probs,
            L=L,
            row=1,
            col=col_idx,
            n_cols=n,
            horizontal_spacing=horizontal_spacing,
            title=title,
            max_patterns=max_patterns,
        )

    # Add colored title annotations above each subplot
    # Calculate subplot domain positions and extract colors
    for col_idx, title in enumerate(titles, start=1):
        if n == 1:
            x_pos = 0.5
        else:
            # Calculate domain for this subplot
            spacing = 0.1 / (n - 1) if n > 1 else 0
            subplot_width = (1.0 - spacing * (n - 1)) / n
            x_start = (col_idx - 1) * (subplot_width + spacing)
            x_end = x_start + subplot_width
            x_pos = (x_start + x_end) / 2
        
        # Add annotation with HTML-formatted text (Plotly supports HTML in annotations)
        fig.add_annotation(
            text=title,
            xref="paper",
            yref="paper",
            x=x_pos,
            y=1.06,  # Position above the subplot
            showarrow=False,
            font=dict(size=14, weight="bold"),
            xanchor="center",
            yanchor="bottom",
        )
    
    # Update layout with axis labels and bottom margin for glyphs
    # Reserve space at bottom for glyphs
    total_glyph_height = L * 0.015 + 0.03  # Approximate height needed
    fig.update_layout(
        margin=dict(b=120, t=120),  # Reserve bottom margin for glyphs
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.14,
            xanchor="center",
            x=0.5,
            entrywidth=25,
            itemwidth=30,
            itemsizing='constant',
        ),
        showlegend=True,
    )
    # --- Annotation outside subplots: orange square + text (no legend)
    # Orange square
    # fig.add_shape(
    #     type="rect",
    #     xref="paper",
    #     yref="paper",
    #     x0=0.02,
    #     x1=0.04,
    #     y0=1.18,
    #     y1=1.205,
    #     fillcolor="orange",
    #     line=dict(width=0),
    #     layer="above",
    # )

    # # Text label next to the square
    # fig.add_annotation(
    #     text="Autoregressive ordering",
    #     xref="paper",
    #     yref="paper",
    #     x=0.045,
    #     y=1.192,
    #     showarrow=False,
    #     xanchor="left",
    #     yanchor="middle",
    #     font=dict(size=12),
    # )
    
    # Add y-axis label to the first subplot
    fig.update_yaxes(title_text=r"$\text{Sampling prob. } q(\mathbf{z}_t | \mathbf{x})$", row=1, col=1)
    
    # Add y-ticks for both subplots
    for col_idx in range(1, n + 1):
        fig.update_yaxes(
            showticklabels=True,
            row=1,
            col=col_idx,
        )
    
    # Add x-axis label to middle subplot
    # Use a single centered x-label below the glyph band (instead of an axis title)
    mid_col = n // 2 + 1 if n > 1 else 1
    fig.update_xaxes(title_text="", row=1, col=mid_col)

    fig.add_annotation(
        text=r"$\text{Masked sequence } \mathbf{z}_t$",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0,
        yshift=-35,
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=14),
    )

    fig.update_layout(
        margin=dict(t=180),   # create headroom
        title=dict[str, str | float | dict[str, int | str]](
            text="Masking patterns for staggered noise matched w/ block size 2, L=4",
            x=0.5,
            y=0.85,            # must be ≤ 1
            xanchor="center",
            yanchor="top",
            font=dict(size=18, weight="bold"),
        ),
    )

    if savepath is not None:
        # For HTML output
        if savepath.endswith('.html'):
            fig.write_html(savepath)
        else:
            # For image output (requires kaleido or orca)
            # Note: plotly requires kaleido or orca for static image export
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
    block_size = 2
    num_blocks = L // block_size
    max_block_size = L
    scale = num_blocks

    # noise = LinearNoise(block_size=block_size,
    #     length=L,
    #     plot_schedule=True)

    # noise = StaggeredNoise(
    #     scale=scale,
    #     length=L,
    #     block_size=L,
    #     plot_schedule=False)

    noise = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=block_size,
        max_block_size=max_block_size,
        length=L,
        plot_schedule=False,
        int_min=0.15)
    # sample_permutation_order(noise, L)
    L = 1024
    d_fixed = 16
    int_mins = [-1.0, 0.1]

    noise_schedules = []
    titles = []

    for int_min in int_mins:
        ns = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=d_fixed,
            max_block_size=L,
            length=L,
            plot_schedule=False,
            int_min=int_min if int_min >= 0.0 else None,
            k=1.0 if int_min < 0.0 else None,
        )
        noise_schedules.append(ns)
    plot_pattern_probabilities_multi_schedule(
        noise_schedules=noise_schedules,
        L=L,
        t_steps=1000,
        masking_trials_per_t=3000,
        max_patterns=None,  # or 16 to keep it uncluttered
        savepath="mask_pattern_probs_by_intmin.png",
    )