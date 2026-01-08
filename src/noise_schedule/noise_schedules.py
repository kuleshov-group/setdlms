from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import numpy as np
import math
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.colors as pc


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
        if length is not None and block_size is not None:
            self.num_blocks = self.length // self.block_size
            self.b = 1 / self.num_blocks
            self.loc = (
                -torch.linspace(0.0, 1 - 1 / self.num_blocks, self.num_blocks)
                .repeat_interleave(self.block_size, dim=-1)
                .flip(0)
            )
        else:
            self.num_blocks = 1
            self.b = 1
            self.loc = torch.tensor([0.0])
    
        print("variance", (self.num_blocks - 1) / (6* self.num_blocks))
        if length is not None and plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/noise_schedules/ar_step/bs{self.block_size}"
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)

    def total_noise(self, t):
        loc = self.loc.to(t.device)
        move_chance = (t + loc) * self.num_blocks
        move_chance = move_chance.clamp(0.0, 1.0)
        return move_chance
        
    def _plot_schedule(self, figdir):
        import numpy as np
        import torch
        import plotly.graph_objects as go
        import plotly.colors as pc

        t = (
            torch.linspace(0, 1, 1000)
            .unsqueeze(-1)
            .repeat(1, self.block_size)
            .repeat_interleave(self.num_blocks, dim=1)
        )

        move_chance = (t + self.loc) * self.num_blocks
        move_chance = move_chance.clamp(0, 1).cpu().numpy()

        x = np.linspace(0, 1, 1000)

        colorscale = pc.sequential.Viridis
        n = max(self.length - 1, 1)

        fig = go.Figure()
        
        active = (move_chance != 0) & (move_chance != 1)
        overlap_per_x = active.sum(axis=1)  # (999,)
        max_overlap = int(overlap_per_x.max())
        avg_overlap = float(overlap_per_x.mean())

        # --- plot all traces ---
        for i in range(self.length):
            u = i / n  # 0..1
            color = pc.sample_colorscale(colorscale, u)[0]

            is_first = (i == 0)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=move_chance[:, i],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=str(i) if self.length <= 8 else None,
                    showlegend=(self.length <= 8),
                    # Shade area under the last curve
                    # fill="tozeroy" if is_first else None,
                    # fillcolor="rgba(68, 1, 84, 0.15)" if is_first else None,  # subtle tint
                )
            )

        # --- AUC for the first index ---
        y_first = move_chance[:, 0]
        auc_first = float(np.trapz(y_first, x))

        # pick a nice point on the first curve to attach an annotation
        x_anno = 0.85
        idx = int(np.clip(np.searchsorted(x, x_anno), 0, len(x) - 1))
        y_anno = float(y_first[idx])

        # fig.add_annotation(
        #     x=x[idx],
        #     y=y_anno,
        #     text=f"AUC(first) = {auc_first:.4f}",
        #     showarrow=True,
        #     arrowhead=2,
        #     ax=40,
        #     ay=-40,
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="rgba(0,0,0,0.2)",
        #     borderwidth=1,
        # )

        # fig.add_annotation(
        #     x=0.01,
        #     y=0.99,
        #     xref="paper",
        #     yref="paper",
        #     xanchor="left",
        #     yanchor="top",
        #     text=f"max # tokens predicted = {max_overlap}, avg # tokens predicted = {auc_first*self.length:.2f}",
        #     showarrow=False,
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="rgba(0,0,0,0.2)",
        #     borderwidth=1,
        # )

        fig.update_layout(
            title=f"<b>Linear noise schedule for block size {self.block_size}</b><br>Max # tokens actively being unmasked: {max_overlap}<br>Expected # tokens actively being unmasked: {auc_first*self.length:.2f}",
            title_x=0.5,
            xaxis_title="t",
            yaxis_title="Mask prob.",
            template="plotly",
            legend_title_text="Token index",
        )

        outpath = f"{figdir}{self.length}schedule.jpg"
        fig.write_image(outpath)

        print("min move chance", move_chance[0].max())
        print("max move chance", move_chance[-1].min())
        print("AUC first index", auc_first)
        print("max block size", max_overlap)
        print("avg block size", avg_overlap)
        print("saved to", outpath)

    def inverse(self, alpha_t):
        return 1 - alpha_t

    def __call__(self, t):
        t = t.to(torch.float32)
        move_chance = self.total_noise(t)
        if self.length is not None and self.block_size is not None:
            num_blocks = self.length // self.block_size
            alpha_t_prime = torch.where(torch.logical_and(t > self.loc, t < self.loc + 1 / num_blocks), -self.num_blocks, 0.)
        else:
            alpha_t_prime = -torch.ones_like(t)
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
        self.b = 1 / self.scale
        self.k = 1.0
        self.init_schedule()
        
        if plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/noise_schedules/staggered/bs{self.block_size}"
            figdir += f'/scale{self.scale[0][0].item():.2f}'
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
        import numpy as np
        import torch
        import plotly.graph_objects as go
        import plotly.colors as pc

        num_blocks = self.length // self.block_size
        t = (
            torch.linspace(0, 1, 1000)
            .unsqueeze(-1)
            .repeat(1, self.block_size)
            .repeat_interleave(num_blocks, dim=1)
        )

        move_chance = self.total_noise(t).cpu().numpy()

        x = np.linspace(0, 1, 1000)

        colorscale = pc.sequential.Viridis
        n = max(self.length - 1, 1)

        fig = go.Figure()
        
        active = (move_chance != 0) & (move_chance != 1)
        overlap_per_x = active.sum(axis=1)  # (999,)
        max_overlap = int(overlap_per_x.max())

        # --- AUC for the first index ---
        y_first = move_chance[:, 0]
        auc_first = float(np.trapz(y_first, x))

        # auc of last curve from 1-b to b
        ts = torch.linspace(1-self.b, self.b, 10000)[:, None]
        y = self.total_noise(ts)
        auc_last = float(np.trapz(y[:, -1].cpu().numpy(), ts[:, -1].cpu().numpy()))
        overlap = (2 * self.b) - 1 - ((self.b / (self.k + 1)) * (2 - (1/self.b))**(self.k+1))
        print("auc of last curve", auc_last)
        print("overlap", overlap)
        # assert auc_last >= self.int_min, f"auc of last curve {auc_last} is less than specified int_min {self.int_min}"
        # print(f"auc of last curve {auc_last} is greater than/equal to specified int_min {self.int_min}")

        # --- plot all traces ---
        for i in range(self.length):
            u = i / n  # 0..1
            color = pc.sample_colorscale(colorscale, u)[0]

            is_first = (i == 0)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=move_chance[:, i],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=str(i+1) if self.length <= 8 else None,
                    showlegend=(self.length <= 8),
                    # Shade area under the last curve
                    # fill="tozeroy" if is_first else None,
                    # fillcolor="rgba(68, 1, 84, 0.15)" if is_first else None,  # subtle tint
                )
            )

        # pick a nice point on the first curve to attach an annotation
        x_anno = 0.85
        idx = int(np.clip(np.searchsorted(x, x_anno), 0, len(x) - 1))
        y_anno = float(y_first[idx])

        # fig.add_annotation(
        #     x=x[idx],
        #     y=y_anno,
        #     text=f"AUC(first) = {auc_first:.4f}",
        #     showarrow=True,
        #     arrowhead=2,
        #     ax=40,
        #     ay=-40,
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="rgba(0,0,0,0.2)",
        #     borderwidth=1,
        # )

        # fig.add_annotation(
        #     x=0.01,
        #     y=0.99,
        #     xref="paper",
        #     yref="paper",
        #     xanchor="left",
        #     yanchor="top",
        #     text=f"max # tokens predicted = {max_overlap}, avg # tokens predicted = {auc_first*self.block_size:.2f}",
        #     showarrow=False,
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="rgba(0,0,0,0.2)",
        #     borderwidth=1,
        # )

        fig.update_layout(
            title=f"<b>Staggered noise schedule</b><br>Max # tokens actively being unmasked: {max_overlap}<br>Expected # tokens actively being unmasked: {auc_first*self.length:.2f}",
            title_x=0.5,
            xaxis_title="t",
            yaxis_title="Mask prob.",
            template="plotly",
            legend_title_text="Token index",
        )

        outpath = f"{figdir}{self.length}schedule.jpg"
        fig.write_image(outpath)

        print("min move chance", move_chance[0].max())
        print("max move chance", move_chance[-1].min())
        print("AUC first index", auc_first)
        print("max block size", max_overlap)
        # print("avg block size", avg_overlap)
        print("saved to", outpath)

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
        batch_size = t.shape[0]
        block_size = block_size if block_size is not None else self.block_size
        num_blocks = to_permute.shape[-1] // block_size
        device = t.device
        if self.scale[0][0] == 1.0:
            to_permute = to_permute.reshape(batch_size, num_blocks, block_size)
            ranking = torch.rand(batch_size, num_blocks, block_size, device=device)
            is_beginning = (to_permute.cumsum(-1) == 0) & (to_permute[:, :, :1] == False)
            if masked_tokens is not None:
                is_beginning = is_beginning | ~masked_tokens
            is_end = ((to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (to_permute[:, :, -1:] == False))

            ranking = torch.where(is_beginning, float('inf'), ranking)
            ranking = torch.where(is_end, float('-inf'), ranking)
            perm_indices = torch.argsort(ranking.cpu(), dim=-1, descending=True, stable=True).to(device)
            perm_indices = perm_indices.reshape(batch_size, num_blocks * block_size)
            # max_deviation = (perm_indices - torch.arange(0, block_size, device=device)[None, :]).abs()
            return perm_indices
        assert num_blocks == 1, "staggered noise schedule only supports one block currently"
        num_total_blocks = t.shape[-1] // block_size * t.shape[0]
        to_permute = to_permute.reshape(-1, block_size)
        is_beginning = (to_permute.cumsum(-1) == 0) & (to_permute[:, :1] == False)
        if masked_tokens is not None:
            is_beginning = is_beginning | ~masked_tokens
        is_end = ((to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (to_permute[:, -1:] == False))

        # sample and sort first-hitting times

        # stable reparametrization, numerically stable for k -> 0
        E = torch.empty(num_total_blocks, block_size, device=device).exponential_()
        loc = self.loc.to(device)
        T = loc[None, :] + self.b * (-torch.expm1(-E / self.k))

        # equivalent, but less numerically stable for k -> 0
        # U = 1 - torch.exp(-E)
        # T_alt = loc[None, :] + self.b * (1 - (1.0 - U)**(1/k))
        # assert (T - T_alt).abs().max() < 1e-4

        # hard constraints
        T = torch.where(is_beginning, float('inf'), T)  # force earliest
        T = torch.where(is_end,      -float('inf'),  T) # force latest
        perms = torch.argsort(T, dim=-1, stable=True, descending=True) # earliest-to-latest
        perms = perms.reshape(batch_size, -1, block_size)
        # max_deviation = (perms - torch.arange(0, block_size, device=device)[None, None, :]).abs()
        perms += torch.arange(num_blocks, device=device)[None, :, None] * block_size
        return perms.reshape(batch_size, -1)

class EaseOutPowerNoise(StaggeredNoise):
    def __init__(self,
            eps=1e-4,
            block_size=1,
            desired_block_size=1,
            max_block_size=1,
            length=1,
            k=None,
            b=None,
            plot_schedule=False,
            int_min=None,
            precision=torch.float64):
        self.int_min = int_min
        if int_min is not None:
            assert int_min >= 0.0 and int_min <= 0.5, f"int_min {int_min} must be between 0.0 and 0.5"
        self.eps = eps
        self.name = "easeoutpower"
        self.length = length
        self.desired_block_size = desired_block_size
        self.max_block_size = max_block_size
        self.precision = precision
        self.init_schedule(block_size, desired_block_size, max_block_size, k, b, int_min, precision)
        print("max active:", (((self.loc + self.b)[-1][-1] - self.loc) > self.eps).sum().item())
        if plot_schedule and self.length < 128:
            figdir = f"/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/noise_schedules/ar_step/bs{self.block_size}"
            figdir += f'/easeoutpower{self.k}b{self.b}'
            figdir += '/'
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            self._plot_schedule(figdir)

    def init_schedule(self, block_size=None, desired_block_size=None, max_block_size=None, k=None, b=None, int_min=None, precision=torch.float64):
        if desired_block_size is None:
            desired_block_size = self.desired_block_size
        if max_block_size is None:
            max_block_size = self.max_block_size
            
        self.block_size = block_size
        desired_num_blocks = self.length / desired_block_size
        desired_area = 1 / (2 * desired_num_blocks)
        frac = (max_block_size - 1) / (self.block_size - 1)
        if int_min is not None:
            int_min = desired_area / 2 # NOTE: HARDCODED
            # int_min = 1 / self.block_size
            b = desired_area / ((2 * desired_area) - int_min)
        if k is not None:
            self.k = k
            self.b = desired_area * (self.k + 1) / self.k
        elif b is not None:
            self.b = b
            self.k = desired_area / (self.b - desired_area)
        else:
            ub_block_size = max_block_size + 1
            frac = (ub_block_size - 1) / (self.block_size - 1)
            lb = frac / (1 + frac)
            denominator = lb - desired_area
            self.b = lb
            self.k = desired_area / denominator
        if int_min is not None:
            overlap = (2 * self.b) - 1 - ((self.b / (self.k + 1)) * (2 - (1/self.b))**(self.k+1))
            assert overlap >= int_min, f"overlap {overlap} is less than int_min {int_min}"
        print(f"k: {self.k}, b: {self.b}")
        assert self.b <= 1.0, f"b {self.b} must be less than or equal to 1.0"
        cur_area = self.k / (self.k + 1) * self.b
        print(f"cur_area: {cur_area}, desired_area: {desired_area}, avg unmasked tokens: {self.block_size * cur_area}")
        
        assert abs(cur_area - desired_area) < 1e-6, f"Current area {cur_area} does not match desired area {desired_area}"
        self.scale = torch.tensor(-1.0)[None, None]
        self.loc = torch.linspace(0., 1.0 - self.b, self.block_size, dtype=precision).flip(0)
        self.loc = self.loc[None, :]

        if self.b < 1.0:
            lower_bound = 1 / self.block_size
            i = self.b * (1 - lower_bound)**(1 / self.k) / (1 - self.b) * (self.block_size - 1) + 1
            print(f"i <= {i}")
            move_chance = self.total_noise(torch.tensor([self.b]))
            print("max active above lower bound", (move_chance >= lower_bound).sum().item())
            print("prob. of masking max_block_size", move_chance[0, -(max_block_size-1)].item())
            print("uniform prob", 1 / self.block_size)

    def total_noise(self, t):
        original_precision = t.dtype
        batch_size = t.shape[0]
        loc = self.loc.to(t.device)
        t = t.to(self.precision)
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, self.block_size)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = (t - loc) / self.b
        eps = torch.finfo(t.dtype).eps

        # numerically stable for k -> 0
        log1m = torch.log1p(-x.clamp(min=0.0, max=1.0 - eps))
        move_chance = -torch.expm1(self.k * log1m)

        # equivalent, but less numerically stable for k -> 0
        # move_chance = 1 - torch.pow(1 - x, self.k)

        move_chance = torch.where(x >= 1.0, 1.0, move_chance)
        move_chance = torch.where(x <= 0.0, 0.0, move_chance)
        return move_chance.reshape(batch_size, -1).to(original_precision)

    def rate_noise(self, t):
        original_precision = t.dtype
        batch_size = t.shape[0]
        loc = self.loc.to(t.device)
        t = t.to(self.precision)
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, self.block_size)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = (t - loc) / self.b
        eps = torch.finfo(t.dtype).eps

        # numerically stable for k -> 0
        log1m = torch.log1p(-x.clamp(min=0.0, max=1.0 - eps))
        alpha_t_prime = - self.k / self.b * torch.exp((self.k - 1.0) * log1m)

        # equivalent, but less numerically stable for k -> 0
        # alpha_t_prime = -self.k / self.b * torch.pow(1 - x, self.k - 1)

        alpha_t_prime = torch.where(x > 1.0, 0.0, alpha_t_prime)
        alpha_t_prime = torch.where(x <= 0.0, 0.0, alpha_t_prime)
        return alpha_t_prime.reshape(batch_size, -1).to(original_precision)

    def __call__(self, t):
        move_chance = self.total_noise(t)
        alpha_t_prime = self.rate_noise(t)
        return 1 - move_chance, alpha_t_prime

    def inverse(self, move_chance):
        original_precision = move_chance.dtype
        if move_chance.ndim > 1 and move_chance.shape[-1] > 1:
            move_chance = move_chance.reshape(-1, self.block_size)
        if move_chance.ndim == 1:
            move_chance = move_chance.unsqueeze(-1)
        batch_size = move_chance.shape[0]
        loc = self.loc.to(move_chance.device)
        eps = torch.finfo(move_chance.dtype).eps

        # numerically stable for k -> 0
        log1m = torch.log1p(-move_chance.clamp(min=0.0, max=1.0 - eps))
        t = loc + self.b * (-torch.expm1((1.0 / self.k) * log1m))

        # equivalent, but less numerically stable for k -> 0
        # t = loc + self.b * (1 - torch.pow(1 - move_chance, 1 / self.k))

        t = torch.clamp(t, min=0.0, max=1.0)
        return t.reshape(batch_size, -1).to(original_precision)
        
    def sample_permutation_order(self, t, to_permute, block_size=None, masked_tokens=None, **kwargs):
        return super().sample_permutation_order(t, to_permute, block_size, masked_tokens, **kwargs)
    def compute_window_size(self):
        return self.block_size
    def _plot_schedule(self, figdir):
        return super()._plot_schedule(figdir)