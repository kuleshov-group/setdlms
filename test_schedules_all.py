from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise, StaggeredNoise
from tqdm import tqdm
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def all_nonzero_patterns_sorted(L: int):
    # all patterns in {0,1}^L except the all-clean one
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

    ordered = all_nonzero_patterns_sorted(L)

    probs_dict = {pat: c / total for pat, c in counts.items()}

    patterns_cond = np.array([list(p) for p in ordered], dtype=int)
    probs_cond = np.array([probs_dict.get(p, 0.0) for p in ordered], dtype=float)
    patterns_cond = patterns_cond[probs_cond > 0]
    if len(patterns_cond) == L:
        probs_cond.fill(1.0 / L)
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
    ymax: float = None,
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
            width=bar_width,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    n_patterns = len(patterns)
    fig.update_xaxes(
        range=[-0.5, n_patterns - 0.5],
        tickmode="linear",
        tick0=0,
        dtick=1,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=row,
        col=col,
    )

    # ymax = max(float(probs.max()) if len(probs) else 1e-8, 1e-8)
    
   # --- Glyphs: draw as layout shapes in paper coords (won't affect y-axis range)
    subplot_idx = (row - 1) * n_cols + col
    xaxis_key = "xaxis" if subplot_idx == 1 else f"xaxis{subplot_idx}"
    xaxis = fig.layout[xaxis_key]
    x_dom0, x_dom1 = xaxis.domain

    band = getattr(fig.layout, "_glyph_band", 0.18)
    yaxis_key = "yaxis" if subplot_idx == 1 else f"yaxis{subplot_idx}"

    d0_plot, d1_plot = fig.layout[yaxis_key].domain
    band = getattr(fig.layout, "_glyph_band", 0.18)
    # glyph band lives directly BELOW the plot domain
    y1_band = d0_plot
    y0_band = d0_plot - band * (d1_plot - d0_plot) / (1.0 - band)


    row_h = (y1_band - y0_band) / max(L, 1)

    square_y = row_h
    fig_w = float(fig.layout.width) if fig.layout.width is not None else 1.0
    fig_h = float(fig.layout.height) if fig.layout.height is not None else 1.0
    square_x = square_y * (fig_h / fig_w)  # compensate for paper-unit aspect
    n_patterns = len(patterns)
    for j, pat in enumerate(patterns):
        # map bar center to paper-x
        # x_center = x_dom0 + (j + 0.5) / n_patterns * (x_dom1 - x_dom0)
        xr = xaxis.range
        x_min, x_max = float(xr[0]), float(xr[1])
        x_span = (x_max - x_min) if (x_max != x_min) else 1.0
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

    # Update subplot layout
    n_patterns = len(patterns)
    fig.update_xaxes(
        type="linear",                 # IMPORTANT: prevent category axis behavior
        range=[-0.5, n_patterns - 0.5],
        autorange=False,               # IMPORTANT: no per-subplot padding
        tickmode="linear",
        tick0=0,
        dtick=1,
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
    figsize=(11, 5),
    savepath: str | None = None,
    device: str | torch.device = "cpu",
):
    """
    One subplot per schedule. Each subplot is a single pattern-probability bar chart.
    """

    n = len(noise_schedules)
    horizontal_spacing = 0.1

    nrows = 2
    ncols = 2

    # Create empty subplot titles - we'll add colored annotations separately
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.18,
        subplot_titles=[""] * (nrows * ncols),
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
    )

    # reserve a bottom band for the glyph grid (shared across subplots)
    fig.layout._glyph_band = min(0.22, 0.045 * L + 0.04)

    band = fig.layout._glyph_band
    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            idx = (r - 1) * ncols + c
            ykey = "yaxis" if idx == 1 else f"yaxis{idx}"
            d0, d1 = fig.layout[ykey].domain
            new_d0 = d0 + band * (d1 - d0)
            fig.update_yaxes(domain=[new_d0, d1], row=r, col=c)

    # --- Clean header layout (no Plotly legend)
    fig.update_layout(
        showlegend=False,
        margin=dict(b=120, t=210),
        plot_bgcolor="white",  # room for: mega title + legend row + subplot titles
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
    m = fig.layout.margin
    ml = m.l or 0
    mr = m.r or 0
    mt = m.t or 0
    mb = m.b or 0
    header_y = 1.2

    # Slightly bigger legend fonts
    LEGEND_LABEL_FS = 15   # was ~14
    LEGEND_SQUARE_FS = 19  # was ~18 for mask, ~20 for clean border
    LEGEND_AR_FS = 15      # was ~14
    d_legend = 0.0125      # keep your spacing from square -> label

    # Compute paper-units-per-pixel for width-based centering
    plot_w = float(fig.layout.width) - (ml + mr)

    def _text_w_paper(txt: str, fs: float) -> float:
        # crude but stable: average glyph ~0.60 * font_size pixels
        return (0.60 * fs * len(txt)) / plot_w

    def _square_w_paper(fs: float) -> float:
        # approximate square glyph width in pixels -> paper units
        return (0.60 * fs) / plot_w

    # Labels (used for width estimates)
    masked_label = "Masked"
    clean_label  = "Clean"

    # Approximate extents in paper units
    sq_w  = _square_w_paper(LEGEND_SQUARE_FS)
    m_tw  = _text_w_paper(masked_label, LEGEND_LABEL_FS)
    c_tw  = _text_w_paper(clean_label,  LEGEND_LABEL_FS)

    item_gap = 0.06  # gap between the two legend items (tweak if desired)

    masked_item_w = sq_w + d_legend + m_tw
    clean_item_w  = sq_w + d_legend + c_tw
    group_w = masked_item_w + item_gap + clean_item_w

    group_left = 0.5 - group_w / 2.0

    # Centers of the square glyphs
    mask_x  = group_left + sq_w / 2.0
    clean_x = group_left + masked_item_w + item_gap + sq_w / 2.0

    # --- Masked: solid square
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

    # --- Clean: outlined square (border + inner fill)
    dx = 0.0 #01
    dy = 0.0

    # Outer (border)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=clean_x, y=header_y,
        text="■",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=LEGEND_SQUARE_FS + 2, color="black"),  # slightly bigger border
    )
    # Inner (fill)
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

    # --- Autoregressive: KEEP orange square position unchanged
    x_ar = -0.01  # unchanged
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
        text="AR prediction",
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=LEGEND_AR_FS),
    )

    titles = []  # Collect titles with pattern counts
    row_idx, col_idx = 1, 1
    all_patterns, all_probs = [], []
    for _, ns in enumerate(noise_schedules, start=1):
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

    for patterns, probs, ns in zip(all_patterns, all_probs, noise_schedules):
        num_masking_patterns = len(patterns)
        num_total_patterns = 2**L - 1
        # title = f'w={ns.b:.1f}, k={ns.k:.1f}'  
        ts = torch.linspace(0, 1, 100000).unsqueeze(-1).repeat(1, L)
        y = ns.total_noise(ts)
        max_overlap = ((y != 0) & (y != 1)).sum(-1).max().item()
        expected_active = max(np.round(ns.k / (ns.k + 1) * ns.b * L, 1), 1.0)
    
        # expected_active = max(float(np.trapz(y[:, 0], ts[:, 0])) * L, 1.0)
        if ns.b <= 1 / L:
            title = f'<b>AR</b><br>Max parallel: {max_overlap:d} token(s)<br>Avg. predicted: {expected_active:.1f} token(s)'
        elif ns.b == 1.0:
            title = f'<b>MDLM</b><br>Max parallel: {max_overlap:d} token(s)<br>Avg. predicted: {expected_active:.1f} token(s)'
        else:
            text_color = "green"
            title = f'<b><span style="color:{text_color};"><i>Soft Block DLM</i></span></b><br>Max parallel: {max_overlap:d} token(s)<br>Avg. predicted: {expected_active:.1f} token(s)'

        # title = f'Unmasking width: {ns.b:.1f}<br><span style="color:{color};">Masking patterns possible: {int(percent_masking_patterns):d}%</span>'    
        titles.append(title)  # Store title for later annotation
        print(row_idx, col_idx)
        plot_pattern_probabilities(
            fig=fig,
            patterns=patterns,
            probs=probs,
            L=L,
            row=row_idx,
            col=col_idx,
            n_cols=ncols,
            horizontal_spacing=horizontal_spacing,
            title=title,
            max_patterns=max_patterns,
            ymax=max([float(probs.max()) if len(probs) else 1e-8 for probs in all_probs]),
        )

        col_idx += 1
        if col_idx > ncols:
            col_idx = 1
            row_idx += 1

    # Add colored title annotations above each subplot
    # Calculate subplot domain positions and extract colors
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
            y=y_top,   # just above each subplot
            showarrow=False,
            font=dict(size=14),
            xanchor="center",
            yanchor="bottom",
        )
    # Update layout with axis labels and bottom margin for glyphs
    # Reserve space at bottom for glyphs
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

    fig.update_yaxes(title_text=r"$\text{Sampling prob. } q(\mathbf{z}_t | \mathbf{x})$", row=2, col=1)
    
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
        yshift=-10,
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=14),
    )
    fig.update_layout(
        margin=dict(l=35, r=15, t=85, b=40, pad=0),
    )

    # fig.update_layout(
    #     margin=dict(t=180),   # create headroom
    #     title=dict[str, str | float | dict[str, int | str]](
    #         text="Masking patterns for staggered noise matched w/ block size 2, L=4",
    #         x=0.5,
    #         y=0.85,            # must be ≤ 1
    #         xanchor="center",
    #         yanchor="top",
    #         font=dict(size=18, weight="bold"),
    #     ),
    # )

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

    # noise = EaseOutPowerNoise(
    #     block_size=L,
    #     desired_block_size=block_size,
    #     max_block_size=max_block_size,
    #     length=L,
    #     plot_schedule=False,
    #     int_min=0.15)
    # sample_permutation_order(noise, L)
    # int_mins = [-1.0, 0.1]
    # desired_block_sizes = [1, 2, 3, 6]
    desired_block_sizes = [1, 2.3, 3, 4]
    # widths = [1/desired_block_sizes[-1], None, None, 1.0]
    # widths = [1/desired_block_sizes[-1], 1/2, 2/3, 1.0]
    widths=[0.25, 0.6, 0.8, 1.0]
    # ks = [None, 0.5, 0.5, None]
    ks = [None] * len(desired_block_sizes)

    noise_schedules = []
    titles = []

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
        max_patterns=None,  # or 16 to keep it uncluttered
        savepath="mask_pattern_probs_by_intmin_all.pdf",
    )