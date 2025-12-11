from abc import ABC, abstractmethod
from typing import Optional
import os
import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import torch

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

    def sample_permutation_order(self, t, to_permute: torch.Tensor, block_size: Optional[int] = None) -> torch.Tensor:
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
        perm_indices = perm_indices.view(batch_size, num_blocks * block_size)
        return perm_indices


# class StaggeredNoise(Noise):
#     def __init__(self, eps=1e-3, scale=1.0, block_size=1, length=1, plot_schedule=False):
#         super().__init__()
#         self.eps = eps
#         self.scale = scale
#         self.block_size = block_size
#         self.length = length
#         self.init_schedule()
        
#         if plot_schedule and self.length < 128:
#             figdir = f"/share/kuleshov/ma2238/dllm-dev-new-/dllm-devnoise_schedules/ar_step/bs{self.block_size}/"
#             figdir += f'scale{self.scale}'
#             figdir += '/'
#             if not os.path.exists(figdir):
#                 os.makedirs(figdir)
#             self._plot_schedule(figdir)

#     def inverse(self, alpha_t):
#         raise NotImplementedError("Inverse function not implemented")

#     def init_schedule(self, scale=None, block_size=None):
#         if block_size is None:
#             block_size = self.block_size
#         if scale is None:
#             scale = self.scale
#         num_blocks = self.length // block_size
#         self.scale = scale
#         self.scale = torch.ones(1, block_size).repeat(1, num_blocks) * scale
#         self.loc =  - torch.linspace(0, 1 - 1 / scale, block_size).flip(0)
#         self.loc = self.loc[None, :].repeat(1, num_blocks)

#         last_token_max_t = 1 / scale
#         self.max_lookahead = (-self.loc < last_token_max_t).sum().item()
    
#     def _plot_schedule(self, figdir):
#         num_blocks = self.length // self.block_size
#         t = torch.linspace(0, 1, 1000).unsqueeze(-1).repeat(1, self.length).repeat_interleave(num_blocks, dim=1)
#         move_chance = self.total_noise(t)
#         move_chance = move_chance.cpu().numpy()

#         colors = n_colors('rgb(68, 1, 84)', 'rgb(253, 231, 37)', self.length, colortype='rgb')
#         fig = go.Figure()
#         for i in range(self.length):
#             fig.add_trace(go.Scatter(
#                 x=np.linspace(0, 1, 1000),
#                 y=move_chance[:, i],
#                 mode='lines',
#                 line=dict(color=colors[i]),
#                 name=str(i) if self.length <= 8 else None
#             ))
#         fig.update_layout(
#             title="Line Plot with Colormap",
#             xaxis_title="X-axis",
#             yaxis_title="Y-axis",
#             showlegend=(self.length <= 8),
#             template="plotly"
#         )
#         plt.savefig(f'{figdir}{self.length}schedule.jpg')
#         print('min move chance', move_chance[0].max())
#         print('max move chance', move_chance[-1].min())
#         print('saved to ', f'{figdir}{self.length}schedule.jpg')

#     def total_noise(self, t):
#         scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
#         t_ = (t + loc) * scale
#         t_ = torch.clamp(t_, 0, 1)
#         return t_

#     def rate_noise(self, t):
#         scale, loc = self.scale.to(t.device)[:, :t.shape[-1]], self.loc.to(t.device)[:, :t.shape[-1]]
#         t_ = (t + loc) * scale
#         t_ = torch.clamp(t_, 0, 1)
#         at_prime = (((t_ != 0) * (t_ != 1)) * scale)

#         # edge case 1: token at min masking prob
#         # at_prime += (t == -loc) * scale

#         # edge case 2: token at max masking prob
#         at_prime += (t == (1/scale - loc)) * scale
#         at_prime = torch.clamp(at_prime, max=scale)
#         return at_prime

#     def __call__(self, t):
#         t = t.to(torch.float32)
#         return self.total_noise(t), self.rate_noise(t)

    
#     def sample_permutation_order(self, t, to_permute: torch.Tensor) -> torch.Tensor:
#         t = t.to(torch.float32)
#         seq_len = t.shape[-1]
#         block_size = self.block_size
#         num_blocks = seq_len // block_size
#         batch_size = t.shape[0]
#         device = t.device
        
#         move_chance = self.total_noise(t)
#         # move_chance = 

#         num_blocks = seq_len // block_size

#         perms = torch.zeros_like(move_chance)
#         mask = torch.ones_like(move_chance)

#         # if we roll T->inf, we mask one token at a time (any-order AR)
#         # 2) for each latent, sample the token most likely to be masked next
#         for i in range(move_chance.shape[0]):
#             current_probs = move_chance[i]
#             current_probs.mul_(mask)

#             # always sample if prob is 1
#             full_mask = current_probs == 1
#             full_mask_idx = (full_mask).any(-1)
#             current_probs = current_probs / current_probs.sum(-1, keepdim=True).clamp(min=1e-8)
#             current_probs[full_mask_idx] = 0
#             current_probs += full_mask
#             sampled_index = torch.multinomial(current_probs, 1, replacement=False)
#             perms[:, i] = sampled_index.squeeze(1)
#             mask.scatter_(1, sampled_index, 0)
#             # mask[torch.arange(mask.shape[0]), sampled_index.squeeze(1)] = 0

#         # 3) FLIP. causal attention from x_T -> x_0
#         perms = perms.view(n, num_blocks, self.block_size)
#         perms += torch.arange(0, seq_len, self.block_size).unsqueeze(0).unsqueeze(-1).to(perms.device)
#         perms = perms.view(n, -1)
#         perms = perms.flip(1)

#         # DEBUG:
#         # max_lookahead = math.ceil(seq_len / self.noise.scale_conf)
#         # assert ((perms - torch.arange(0, x0.shape[1]).unsqueeze(0).to(perms.device)).abs() < max_lookahead).all()

#         # bos will always be the first index.
#         # 5) move zero idx to the first position, shift
#         zero_idx = (perms != 0)
#         perms = perms[zero_idx].view(n, perms.shape[1]-1)
#         perms = torch.cat((torch.zeros_like(perms[:, 0].unsqueeze(1)), perms), dim=1)
#         perms = perms.to(t.device).to(torch.long)

#         return perms