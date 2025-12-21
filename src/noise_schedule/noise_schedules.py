from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import torch
import math
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
    def __init__(self, block_size: Optional[int] = None):
        super().__init__()
        self.name = "linear"

    def inverse(self, alpha_t):
        return 1 - alpha_t

    def __call__(self, t):
        t = t.to(torch.float32)
        alpha_t_prime = -torch.ones_like(t)
        move_chance = t
        return 1 - move_chance, alpha_t_prime

    def sample_permutation_order(self, t, to_permute: torch.Tensor, block_size: Optional[int] = None, **kwargs: Any) -> torch.Tensor:
        t = t.to(torch.float32)
        seq_len = t.shape[-1]
        block_size = block_size
        num_blocks = seq_len // block_size
        batch_size = t.shape[0]
        device = t.device

        to_permute = to_permute.reshape(batch_size, num_blocks, block_size)
        ranking = torch.rand(batch_size, num_blocks, block_size, device=device)
        position_indices = torch.arange(block_size, device=device)[None, None, :]
        is_beginning = (to_permute.cumsum(-1) == 0) & (to_permute[:, :, :1] == False)
        is_end = ((to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (to_permute[:, :, -1:] == False))

        ranking = torch.where(is_beginning, float('inf'), ranking)
        ranking = torch.where(is_end, float('-inf'), ranking)
        perm_indices = torch.argsort(ranking.cpu(), dim=-1, descending=True, stable=True).to(device)
        perm_indices = perm_indices.reshape(batch_size, num_blocks * block_size)
        return perm_indices


class StaggeredNoise(Noise):
    def __init__(self, eps=1e-3, scale=1.0, block_size=1, length=1, plot_schedule=False):
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.block_size = block_size
        self.length = length
        self.init_schedule()
        
        if plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new-/dllm-devnoise_schedules/ar_step/bs{self.block_size}/"
            figdir += f'scale{self.scale}'
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)

    def inverse(self, alpha_t):
        raise NotImplementedError("Inverse function not implemented")

    def compute_window_size(self):
        # return (-self.loc < (1/self.scale[0][0])).sum().item()
        if self.scale[0][0] == 1:
            return self.block_size
        max_parallel = math.ceil((2 - self.scale[0][0] - self.block_size) / (1 - self.scale[0][0])) - 1
        return min(max_parallel, self.block_size)

    def init_schedule(self, scale=None, block_size=None):
        if block_size is None:
            block_size = self.block_size
        if scale is None:
            scale = self.scale
        num_blocks = self.length // block_size
        self.scale = scale
        self.scale = torch.ones(1, block_size) * scale
        self.loc =  - torch.linspace(0, 1 - 1 / scale, block_size).flip(0)
        self.loc = self.loc[None, :]
    
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
        if self.block_size == 1:
            return torch.ones_like(t)
        scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
        batch_size = t.shape[0]
        t = t.reshape(-1, self.block_size)
        move_chance = (t + loc) * scale
        move_chance = torch.clamp(move_chance, 0, 1)
        return move_chance.reshape(batch_size, -1)

    def inverse_noise(self, move_chance):
        scale, loc = self.scale.to(move_chance.device)[:, :move_chance.shape[-1]], self.loc.to(move_chance.device)[:, :move_chance.shape[-1]]
        batch_size = move_chance.shape[0]
        move_chance = move_chance.reshape(-1, self.block_size)
        t = (move_chance / scale) - loc
        t = torch.clamp(t, 0, 1)
        return t.reshape(batch_size, -1)

    def rate_noise(self, t):
        scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
        batch_size = t.shape[0]
        t = t.reshape(-1, self.block_size)
        move_chance = (t + loc) * scale
        move_chance = torch.clamp(move_chance, 0, 1)
        at_prime = (((move_chance != 0) * (move_chance != 1)) * scale)

        # edge case: token at max masking prob
        at_prime += (t == (1/scale - loc)) * scale
        at_prime = torch.clamp(at_prime, max=scale)
        return at_prime.reshape(batch_size, -1)

    def __call__(self, t):
        t = t.to(torch.float32)
        return 1 - self.total_noise(t), - self.rate_noise(t)

    def sample_permutation_order(
        self,
        t,
        to_permute: torch.Tensor,
        block_size: Optional[int] = None,
        masked_tokens: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = to_permute.shape[0]
        block_size = block_size if block_size is not None else self.block_size

        num_total_blocks = t.shape[-1] // block_size * t.shape[0]
        num_blocks = to_permute.shape[-1] // block_size
        t = torch.ones(num_total_blocks, block_size)

        to_permute = to_permute.reshape(-1, block_size)
        is_beginning = (to_permute.cumsum(-1) == 0) & (to_permute[:, :1] == False)
        is_end = ((to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (to_permute[:, -1:] == False))

        mask = torch.ones(num_total_blocks, block_size).to(to_permute.device)

        # 1) Start with full noise
        ranking = torch.full((num_total_blocks, block_size), fill_value=-1, dtype=torch.long).to(to_permute.device)
        unmask_chance = 1 - self.total_noise(t - self.eps).to(to_permute.device)
        
        # 2) Sample next-hitting time for every token
        uniform_samp = torch.rand(num_total_blocks).to(to_permute.device)
        mask_chance = 1 - (uniform_samp.unsqueeze(-1) * (1 - unmask_chance))
        t = self.inverse_noise(mask_chance).max(dim=-1).values
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = t.repeat(1, block_size)
        if masked_tokens is not None:
            masked_tokens = masked_tokens.reshape(-1, block_size)
        for i in range(block_size):
            if -1 not in ranking:
                break
            if i == block_size - 1:
                ranking[ranking == -1] = i
                break
            # Get unmasking probs for the next-hitting time
            unmask_chance = 1 - self.total_noise(t).to(to_permute.device)

            # 3) Sample a token
            unmask_chance_filtered = torch.where(is_beginning, 1.0, unmask_chance)
            unmask_chance_filtered = torch.where(is_end, 0.0, unmask_chance_filtered)
            unmask_chance_filtered *= mask

            full_unmask_chance_mask = (unmask_chance_filtered == 1.0) * (mask == 1.0)
            unmask_chance_filtered[(full_unmask_chance_mask).any(dim=-1)] = 0.0

            # 4) During EVAL, use sampled x_t to determine the unmasking order (for CACHING)
            if masked_tokens is not None:
                # If clean tokens aren't already ranked, set the unmasking chance for masked tokens to 0
                for j in range(num_total_blocks):
                    clean_mask = ~masked_tokens[j]
                    remaining_clean_tokens_rank = ~torch.isin(clean_mask.nonzero().squeeze(-1), ranking[j])
                    if remaining_clean_tokens_rank.any():
                        unmask_chance_filtered[j][masked_tokens[j]] = 0.0

            # For tie-breakers where unmasking chance is 1.0, prioritize leftmost token
            unmask_chance_filtered[(full_unmask_chance_mask).any(dim=-1)] = ((full_unmask_chance_mask.cumsum(-1) == 1) * (unmask_chance_filtered == 1.0))[(full_unmask_chance_mask).any(dim=-1)].float()
            if (unmask_chance_filtered.sum(-1) <= 0).any():
                unmask_chance_filtered[unmask_chance_filtered <= 0] = ((mask.cumsum(-1) == 1) * mask > 0)[unmask_chance_filtered <= 0].float()
            
            samp = torch.multinomial(unmask_chance_filtered, 1)

            ranking[:, i] = samp.squeeze(-1)
            mask.scatter_(1, samp, 0)

            if i == block_size - 1:
                break
            # 2) Sample next-hitting time for every token, based on current unmasking probs
            uniform_samp = torch.rand(num_total_blocks).to(to_permute.device)
            mask_chance = 1 - (uniform_samp.unsqueeze(-1) * (1 - unmask_chance))
            t = (self.inverse_noise(mask_chance) * mask).max(dim=-1).values
            t = t.unsqueeze(-1).repeat(1, block_size)

        perms = ranking.reshape(batch_size, -1, block_size)
        perms += torch.arange(num_blocks, device=perms.device)[None, :, None] * block_size

        # max_lookahead = math.ceil((block_size - 1) / (self.scale[0][0] - 1)) + 1
        # max_deviation = (perms - torch.arange(0, block_size)[None, None, :]).abs()
        # assert (max_deviation < max_lookahead).all()
        
        perms = perms.reshape(batch_size, -1)
        return perms