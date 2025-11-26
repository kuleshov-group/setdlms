from abc import ABC, abstractmethod
from typing import Optional
import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch

import torch


class Noise(ABC):
    """
    Baseline forward method to get noise parameters at a timestep
    """

    def __call__(
        self, t: torch.Tensor | float
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        # Assume time goes from 0 to 1
        pass

    @abstractmethod
    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """
        Inverse function to compute the timestep t from the noise schedule param.
        """
        raise NotImplementedError("Inverse function not implemented")


class CosineNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.name = "cosine"

    def __call__(self, t):
        t = t.to(torch.float32)
        cos = -(1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = -(1 - self.eps) * torch.sin(t * torch.pi / 2)
        move_chance = cos + 1
        alpha_t_prime = sin * torch.pi / 2
        return 1 - move_chance, alpha_t_prime


class ExponentialNoise(Noise):
    def __init__(self, exp=2, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.exp = exp
        self.name = f"exp_{exp}"

    def __call__(self, t):
        t = t.to(torch.float32)
        move_chance = torch.pow(t, self.exp)
        move_chance = torch.clamp(move_chance, min=self.eps)
        alpha_t_prime = -self.exp * torch.pow(t, self.exp - 1)
        return alpha_t_prime, 1 - move_chance


class LogarithmicNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.name = "logarithmic"

    def __call__(self, t):
        t = t.to(torch.float32)
        move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
        alpha_t_prime = -1 / (torch.log(torch.tensor(2.0)) * (1 + t))
        return 1 - move_chance, alpha_t_prime


class LinearNoise(Noise):
    def __init__(self):
        super().__init__()
        self.name = "linear"

    def inverse(self, alpha_t):
        return 1 - alpha_t

    def __call__(self, t):
        t = t.to(torch.float32)
        alpha_t_prime = -torch.ones_like(t)
        move_chance = t
        return 1 - move_chance, alpha_t_prime

    def sample_permutation_order(self, batch_size: int, seq_len: int, block_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        num_blocks = seq_len // block_size
        ranking = torch.rand(batch_size, n_blocks, block_size, device=device)
        ranking = torch.where(to_permute, ranking, float('inf'))
        perm_indices = torch.argsort(ranking.cpu(), dim=-1, descending=True, stable=True).to(device)
        return perm_indices


class StaggeredNoise(Noise):
    def __init__(self, eps=1e-3, scale_conf=1.0, block_size=1, length=1, plot_schedule=False):
        super().__init__()
        self.eps = eps
        self.scale_conf = scale_conf
        self.block_size = block_size
        self.length = length
        self.init_schedule()
        
        if plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new-/dllm-devnoise_schedules/ar_step/bs{self.block_size}/"
            figdir += f'scale{self.scale_conf}'
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)

    def inverse(self, alpha_t):
        raise NotImplementedError("Inverse function not implemented")

    def init_schedule(self, scale_conf=None, block_size=None):
        if block_size is None:
            block_size = self.block_size
        if scale_conf is None:
            scale_conf = self.scale_conf
        num_blocks = self.length // block_size
        self.scale_conf = scale_conf
        self.scale = torch.ones(1, block_size).repeat(1, num_blocks) * scale_conf
        self.loc =  - torch.linspace(0, 1 - 1 / scale_conf, block_size).flip(0)
        self.loc = self.loc[None, :].repeat(1, num_blocks)

        last_token_max_t = 1 / scale_conf
        self.max_lookahead = (-self.loc < last_token_max_t).sum().item()
    
    def _plot_schedule(self, figdir):
        num_blocks = self.length // self.block_size
        t = torch.linspace(0, 1, 1000).unsqueeze(-1).repeat(1, self.length).repeat_interleave(num_blocks, dim=1)
        move_chance = self.total_noise(t)
        move_chance = move_chance.cpu().numpy()

        colors = n_colors('rgb(68, 1, 84)', 'rgb(253, 231, 37)', self.length, colortype='rgb')
        fig = go.Figure()
        for i in range(self.length):
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, 1000),
                y=move_chance[:, i],
                mode='lines',
                line=dict(color=colors[i]),
                name=str(i) if self.length <= 8 else None
            ))
        fig.update_layout(
            title="Line Plot with Colormap",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            showlegend=(self.length <= 8),
            template="plotly"
        )
        plt.savefig(f'{figdir}{self.length}schedule.jpg')
        print('min move chance', move_chance[0].max())
        print('max move chance', move_chance[-1].min())
        print('saved to ', f'{figdir}{self.length}schedule.jpg')

    def total_noise(self, t):
        scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
        t_ = (t + loc) * scale
        t_ = torch.clamp(t_, 0, 1)
        return t_

    def rate_noise(self, t):
        scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
        t_ = (t + loc) * scale
        t_ = torch.clamp(t_, 0, 1)
        at_prime = (((t_ != 0) * (t_ != 1)) * scale)

        # edge case 1: token at min masking prob
        # at_prime += (t == -loc) * scale

        # edge case 2: token at max masking prob
        at_prime += (t == (1/scale - loc)) * scale
        at_prime = torch.clamp(at_prime, max=scale)
        return at_prime

    def __call__(self, t):
        t = t.to(torch.float32)
        return self.total_noise(t), self.rate_noise(t)

    
    def sample_permutation_order(self, batch_size: int, seq_len: int, block_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError("Sample permutation order not implemented")
        num_blocks = seq_len // block_size
        ranking = torch.rand(batch_size, n_blocks, block_size, device=device)
        ranking = torch.where(to_permute, ranking, float('inf'))
        perm_indices = torch.argsort(ranking.cpu(), dim=-1, descending=True, stable=True).to(device)
        return perm_indices