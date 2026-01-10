from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise, StaggeredNoise
from tqdm import tqdm
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    for ti in tqdm(range(t_min_index, t_max_index), desc="Simulating pattern probabilities across t"):
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

import numpy as np
import plotly.graph_objects as go

def plot_inf_budget_barplot_plotly(
    block_sizes,
    empirical_means,
    empirical_stds,
    theoretical_values,
    title="Inference prediction budget vs block size",
    x_title="Block size",
    y_title="Inference prediction budget",
    t_steps: int = 1000,
    masking_trials_per_t: int = 2000,
):
    """
    Blue bars: empirical_means with vertical error bars (empirical_stds)
    Red dots: theoretical_values centered in each bar (same x)
    """
    # Ensure ascending by block size
    order = np.argsort(np.array(block_sizes))
    block_sizes = np.array(block_sizes)[order]
    empirical_means = np.array(empirical_means)[order]
    empirical_stds = np.array(empirical_stds)[order]
    theoretical_values = np.array(theoretical_values)[order]

    fig = go.Figure()
    total_samples = t_steps * masking_trials_per_t
    total_samples_text = int(total_samples // 1e6)

    # Empirical bars + error bars
    fig.add_trace(
        go.Bar(
            x=block_sizes,
            y=empirical_means,
            name=f"Empirical (total samples={total_samples_text}M)",
            marker=dict(color="royalblue"),
            # error_y=dict(type="data", array=empiv rical_stds, visible=True),
        )
    )

    # Theoretical red dots (centered at same x)
    fig.add_trace(
        go.Scatter(
            x=block_sizes,
            y=theoretical_values,
            mode="markers",
            name="Theoretical",
            marker=dict(color="red", size=10, symbol="circle"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        barmode="overlay",
        template="plotly_white",
    )
    fig.update_xaxes(type="category")  # keeps bars evenly spaced even if sizes jump (2,4,8,...)

    return fig
def plot_pattern_probabilities_multi_schedule(
    noise_schedules,
    block_sizes,
    L: int,
    t_steps: int = 1000,
    masking_trials_per_t: int = 2000,
    device: str | torch.device = "cpu",
    x_title: str = "Desired block size",
    y_title: str = "Budget (mean # predicted)",
    title: str = "Inference prediction budget vs block size",
):
    empirical_means, empirical_stds, theoretical_vals = [], [], []

    for ns, bs in zip(noise_schedules, block_sizes):
        _, _, num_predicted = simulate_pattern_probs_across_t(
            noise=ns,
            L=L,
            t_steps=t_steps,
            masking_trials_per_t=masking_trials_per_t,
            device=device,
        )

        empirical_means.append(float(np.mean(num_predicted)))
        empirical_stds.append(float(np.std(num_predicted)))
        theoretical_vals.append(float(ns.compute_inf_budget()))

        print(f"block_size={bs:>4} | empirical: {empirical_means[-1]:.4f} +/- {empirical_stds[-1]:.4f}"
              f" | theoretical: {theoretical_vals[-1]:.4f}")

    fig = plot_inf_budget_barplot_plotly(
        block_sizes=block_sizes,
        empirical_means=empirical_means,
        empirical_stds=empirical_stds,
        theoretical_values=theoretical_vals,
        title=title,
        x_title=x_title,
        y_title=y_title,
        t_steps=t_steps,
        masking_trials_per_t=masking_trials_per_t,
    )
    return fig
if __name__ == "__main__":
    L = 128
    block_size = 16
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
    block_sizes = [2, 4, 8, 16, 32, 64, 128]

    noise_schedules = []
    titles = []

    for block_size in block_sizes:
        ns = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=block_size,
            max_block_size=L,
            length=L,
            plot_schedule=False,
            int_min=0.1,
        )
        noise_schedules.append(ns)
    fig = plot_pattern_probabilities_multi_schedule(
        noise_schedules=noise_schedules,
        block_sizes=block_sizes,
        L=L,
        t_steps=1000,
        masking_trials_per_t=3000,
        device="cpu",
        x_title="Matched block size",
        y_title="Inference prediction budget",
        title="Soft block diffusion: Matched block size vs inference prediction budget",
    )
    fig.write_image("inf_budget_by_block_size.png")  #
    print("saved figure to inf_budget_by_block_size.png")