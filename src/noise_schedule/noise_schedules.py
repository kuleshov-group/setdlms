from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import numpy as np
import math
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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
    def __init__(self, block_size: Optional[int] = None, length: Optional[int] = None, plot_schedule: Optional[bool] = False):
        super().__init__()
        self.name = "linear"
        self.block_size = block_size
        self.length = length
        self.plot_schedule = plot_schedule
        if length is not None and plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new-/dllm-devnoise_schedules/ar_step/bs{self.block_size}/"
            figdir += f'scale{self.scale}'
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)
        
    def _plot_schedule(self, figdir):
        num_blocks = self.length // self.block_size
        t = torch.linspace(0, 1, 1000).unsqueeze(-1).repeat(1, self.length).repeat_interleave(num_blocks, dim=1)
        move_chance = self.total_noise(t)
        move_chance = move_chance.cpu().numpy()

        # scale by num_blocks
        move_chance = move_chance * num_blocks

        # offset by block index
        move_chance = move_chance + torch.arange(num_blocks, device=move_chance.device)[None, :] / num_blocks

        fig = go.Figure()
        for i in range(self.length):
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, 1000),
                y=move_chance[:, i],
                mode='lines',
                name=str(i) if self.length <= 8 else None
            ))
        fig.update_layout(
            title="Line Plot with Colormap",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            showlegend=(self.length <= 8),
            template="plotly"
        )
        fig.write_image(f'{figdir}{self.length}schedule.jpg')
        print('min move chance', move_chance[0].max())
        print('max move chance', move_chance[-1].min())
        print('saved to ', f'{figdir}{self.length}schedule.jpg')

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

        fig = go.Figure()
        for i in range(self.length):
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, 1000),
                y=move_chance[:, i],
                mode='lines',
                name=str(i) if self.length <= 8 else None
            ))
        fig.update_layout(
            title="Line Plot with Colormap",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            showlegend=(self.length <= 8),
            template="plotly"
        )
        fig.write_image(f'{figdir}{self.length}schedule.jpg')
        print('min move chance', move_chance[0].max())
        print('max move chance', move_chance[-1].min())
        print('saved to ', f'{figdir}{self.length}schedule.jpg')

    def total_noise(self, t):
        if self.block_size == 1:
            return torch.ones_like(t)
        scale, loc = self.scale.to(t.device), self.loc.to(t.device)
        batch_size = t.shape[0]
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, self.block_size)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        move_chance = (t + loc) * scale
        move_chance = torch.clamp(move_chance, 0, 1)
        return move_chance.reshape(batch_size, -1)

    def inverse(self, move_chance):
        scale, loc = self.scale.to(move_chance.device), self.loc.to(move_chance.device)
        batch_size = move_chance.shape[0]
        if move_chance.ndim > 1 and move_chance.shape[-1] > 1:
            move_chance = move_chance.reshape(-1, self.block_size)
        if move_chance.ndim == 1:
            move_chance = move_chance.unsqueeze(-1)
        t = (move_chance / scale) - loc
        t = torch.clamp(t, 0, 1)
        return t.reshape(batch_size, -1)

    def rate_noise(self, t):
        scale, loc = self.scale.to(t.device), self.loc.to(t.device)
        batch_size = t.shape[0]
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, self.block_size)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
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
        if self.scale[0][0] == 1.0:
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
            max_deviation = (perm_indices - torch.arange(0, block_size, device=device)[None, :]).abs()
            return perm_indices

        batch_size = to_permute.shape[0]
        block_size = block_size if block_size is not None else self.block_size
        device = to_permute.device

        num_total_blocks = t.shape[-1] // block_size * t.shape[0]
        num_blocks = to_permute.shape[-1] // block_size

        to_permute = to_permute.reshape(-1, block_size)
        is_beginning = (to_permute.cumsum(-1) == 0) & (to_permute[:, :1] == False)
        is_end = ((to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (to_permute[:, -1:] == False))

        if masked_tokens is not None:
            masked_tokens = masked_tokens.reshape(-1, block_size)

        # Start with full noise
        t = torch.ones(num_total_blocks, 1, device=device)

        # Per-token masking probabilities
        mask_chance = self.total_noise(t)
        
        # Pre-allocate random numbers for entire loop (use float32 for efficiency)
        uniform_samp = torch.rand(num_total_blocks, self.block_size, self.block_size, device=device, dtype=torch.float32)
        g = -torch.empty(num_total_blocks, self.block_size, self.block_size, device=device, dtype=torch.float32).exponential_().log()

        # Sample next-hitting time for every token, based on current unmasking probs
        unmask_chance_list = []

        # (B x L/S) x S
        perms = torch.full((num_total_blocks, block_size), fill_value=-1, dtype=torch.long, device=device)
        mask = torch.full((num_total_blocks, block_size), fill_value=True, dtype=torch.bool, device=device)
        window_size = self.compute_window_size()

        def gumbel_noise(logits, g):
            g = torch.where(logits == 0, -float('inf'), g)
            g = torch.where(logits == 1, float('inf'), g)
            return (logits + g)

        for i in range(block_size):
            # Get next-hitting time from available tokens
            t = (self.inverse(uniform_samp[:, i] * mask_chance) * mask).max(dim=-1).values
            unmask_chance = 1 - self.total_noise(t)

            # Hard-code unmasking probability for prompt/pad tokens
            unmask_chance_filtered = torch.where(is_beginning, 1.0, unmask_chance)
            unmask_chance_filtered = torch.where(is_end, 0.0, unmask_chance_filtered)

            # Ignore previously sampled tokens
            unmask_chance_filtered *= mask

            # For tie-breakers where unmasking chance is 1.0, prioritize leftmost token
            full_unmask_chance_mask = (unmask_chance_filtered == 1.0)
            unmask_chance_filtered[(full_unmask_chance_mask).any(dim=-1)] = ((full_unmask_chance_mask.cumsum(-1) == 1) * (unmask_chance_filtered == 1.0))[(full_unmask_chance_mask).any(dim=-1)].float()
            if (unmask_chance_filtered.sum(-1) <= 0).any():
                unmask_chance_filtered[unmask_chance_filtered <= 0] = ((mask.cumsum(-1) == 1) * mask > 0)[unmask_chance_filtered <= 0].float()

            # Sample a token
            perms[:, i] = gumbel_noise(unmask_chance_filtered, g[:, i]).argmax(dim=-1)
            mask.scatter_(1, perms[:, i].unsqueeze(-1), False)
            
            if i == block_size - 1:
                break
            mask_chance = self.total_noise(t)
    

        perms = perms.reshape(batch_size, -1, block_size)
        perms += torch.arange(num_blocks, device=device)[None, :, None] * block_size
        perms = perms.reshape(batch_size, -1)
        max_deviation = (perms - torch.arange(0, block_size, device=device)[None, :]).abs()

        if max_deviation.max() > window_size:
            raise ValueError(f'Window size violation at final step', max_deviation.max(), window_size, self.scale[0][0])

        return perms

class EaseOutPowerNoise(StaggeredNoise):
    def __init__(self, eps=1e-3, block_size=1, desired_block_size=1, max_block_size=1, length=1, plot_schedule=False):
        super().__init__()
        self.eps = eps
        self.name = "easeoutpower"
        self.block_size = block_size
        self.b = max_block_size / length
        desired_num_blocks = length / desired_block_size
        desired_area = (desired_num_blocks / 2) * (1 / desired_num_blocks)**2 # using block diffusion slope
        self.k = desired_area / (self.b - desired_area)
        cur_area = self.k / (self.k + 1) * self.b
        assert abs(cur_area - desired_area) < 1e-6, f"Current area {cur_area} does not match desired area {desired_area}"
        self.scale = torch.tensor(self.k)[None, None]
        self.length = length
        self.loc = torch.linspace(0., 1.0 - self.b, self.length).flip(0)
        self.loc = self.loc[None, :]
        if plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new-/dllm-devnoise_schedules/ar_step/bs{self.block_size}/"
            figdir += f'easeoutpower{self.k}'
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)

    def total_noise(self, t):
        loc = self.loc.to(t.device).to(t.dtype)
        t = t.to(torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = (t - loc) / self.b
        move_chance = 1 - torch.pow(1 - x, self.k)
        move_chance = torch.where(x > 1, 1, move_chance)
        move_chance = torch.where(x < 0, 0, move_chance)
        move_chance = torch.clamp(move_chance, min=0, max=1)
        return move_chance


    def __call__(self, t):
        t = t.to(torch.float32)
        loc = self.loc.to(t.device).to(t.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = (t - loc) / self.b
        move_chance = self.total_noise(t)
        alpha_t_prime = -self.k / self.b * torch.pow(1 - x, self.k - 1)
        return 1 - move_chance, alpha_t_prime

    def inverse(self, p):
        loc = self.loc.to(p.device).to(p.dtype)
        t = loc + self.b * (1 - torch.pow(1 - p, 1 / self.k))
        t = torch.clamp(t, min=0, max=1)
        return t
        
    def sample_permutation_order(self, t, to_permute, block_size=None, masked_tokens=None, **kwargs):
        return super().sample_permutation_order(t, to_permute, block_size, masked_tokens, **kwargs)
    def compute_window_size(self):
        return math.floor(self.b * self.length)
    def init_schedule(self, scale=None, block_size=None):
        return
    def _plot_schedule(self, figdir):
        return super()._plot_schedule(figdir)