from abc import ABC
from typing import Any, Optional

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


class LinearNoise(Noise):
    def __init__(
        self,
        block_size: Optional[int] = None,
        length: Optional[int] = None,
    ):
        super().__init__()
        self.name = "linear"
        self.block_size = block_size
        self.length = length
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

    def compute_window_size(self):
        return self.block_size

    def total_noise(self, t):
        loc = self.loc.to(t.device)
        move_chance = (t + loc) * self.num_blocks
        move_chance = move_chance.clamp(0.0, 1.0)
        return move_chance

    def __call__(self, t):
        t = t.to(torch.float32)
        move_chance = self.total_noise(t)
        if self.length is not None and self.block_size is not None:
            num_blocks = self.length // self.block_size
            loc = self.loc.to(t.device)
            alpha_t_prime = torch.where(
                torch.logical_and(t > loc, t < loc + 1 / num_blocks),
                -self.num_blocks,
                0.0,
            )
        else:
            alpha_t_prime = -torch.ones_like(t)
        return 1 - move_chance, alpha_t_prime

    def compute_first_hitting_times(
        self, batch_size, length, device, dtype=torch.float64
    ):
        timesteps = torch.FloatTensor([1.0]).to(device).repeat(batch_size, 1)
        for i in range(length, 0, -1):
            eps = torch.finfo(dtype).tiny
            u = torch.rand(batch_size, device=device).clamp_min(eps)
            next_t = timesteps[:, -1] * torch.exp(torch.log(u) / i)
            timesteps = torch.cat((timesteps, next_t[:, None]), dim=1)
        return timesteps[:, 1:].to(device, dtype=dtype)  # type: ignore

    def sample_permutation_order(
        self,
        t,
        to_permute: torch.Tensor,
        block_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        t = t.to(torch.float32)
        seq_len = t.shape[-1]
        block_size = block_size
        num_blocks = seq_len // block_size
        batch_size = t.shape[0]
        device = t.device

        to_permute = to_permute.reshape(batch_size, num_blocks, block_size)
        ranking = torch.rand(batch_size, num_blocks, block_size, device=device)
        is_beginning = (to_permute.cumsum(-1) == 0) & (~to_permute[:, :, :1])
        is_end = (to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (
            ~to_permute[:, :, -1:]
        )

        ranking = torch.where(is_beginning, float("inf"), ranking)
        ranking = torch.where(is_end, float("-inf"), ranking)
        perm_indices = torch.argsort(
            ranking.cpu(), dim=-1, descending=True, stable=True
        ).to(device)
        perm_indices = perm_indices.reshape(batch_size, num_blocks * block_size)
        return perm_indices


class StaggeredNoise(Noise):
    def __init__(
        self,
        eps=1e-4,
        block_size=1,
        desired_block_size=1,
        max_block_size=1,
        length=1,
        k=None,
        b=None,
        precision=torch.float64,
    ):
        super().__init__()
        assert max_block_size <= block_size, (
            f"max_block_size {max_block_size} must be less than or equal to "
            f"block_size {block_size}"
        )
        self.eps = eps
        self.name = "staggered"
        self.length = length
        self.desired_block_size = desired_block_size
        self.max_block_size = max_block_size
        self.precision = precision
        self.init_schedule(
            block_size, desired_block_size, max_block_size, k, b, precision
        )

    def init_schedule(
        self,
        block_size=None,
        desired_block_size=None,
        max_block_size=None,
        k=None,
        b=None,
        precision=torch.float64,
    ):
        if desired_block_size is None:
            desired_block_size = self.desired_block_size
        if max_block_size is None:
            max_block_size = self.max_block_size

        self.block_size = block_size
        desired_num_blocks = self.length / desired_block_size
        desired_area = 1 / (2 * desired_num_blocks)

        if k is not None:
            self.k = k
            self.b = desired_area * (self.k + 1) / self.k
        elif b is not None:
            self.b = b
            self.k = desired_area / (self.b - desired_area)
        elif desired_block_size not in [1, self.max_block_size]:
            self.b = self.max_block_size / (self.block_size + self.max_block_size - 1)
            self.k = desired_area / (self.b - desired_area)
        else:
            self.k = 1.0
            self.b = desired_area * (self.k + 1) / self.k

        assert self.b <= 1.0, f"b {self.b} must be less than or equal to 1.0"
        cur_area = self.k / (self.k + 1) * self.b
        assert abs(cur_area - desired_area) < 1e-6, (
            f"Current area {cur_area} does not match desired area {desired_area}"
        )
        self.scale = torch.tensor(-1.0)[None, None]
        self.loc = torch.linspace(
            0.0, 1.0 - self.b, self.block_size, dtype=precision
        ).flip(0)
        self.loc = self.loc[None, :]

    def total_noise(self, t):
        block_size = min(self.block_size, t.shape[-1])
        original_precision = t.dtype
        batch_size = t.shape[0]
        loc = self.loc.to(t.device)[:, -block_size:]
        t = t.to(self.precision)
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, block_size)
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
        if self.desired_block_size == 1:
            move_chance = torch.where(move_chance > 0.0, 1.0, move_chance)
        return move_chance.reshape(batch_size, -1).to(original_precision)

    def rate_noise(self, t):
        original_precision = t.dtype
        batch_size = t.shape[0]
        block_size = min(self.block_size, t.shape[-1])
        loc = self.loc.to(t.device)[:, -block_size:]
        t = t.to(self.precision)
        if t.ndim > 1 and t.shape[-1] > 1:
            t = t.reshape(-1, block_size)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = (t - loc) / self.b
        eps = torch.finfo(t.dtype).eps

        # numerically stable for k -> 0
        log1m = torch.log1p(-x.clamp(min=0.0, max=1.0 - eps))
        alpha_t_prime = -self.k / self.b * torch.exp((self.k - 1.0) * log1m)

        # equivalent, but less numerically stable for k -> 0
        # alpha_t_prime = -self.k / self.b * torch.pow(1 - x, self.k - 1)

        alpha_t_prime = torch.where(x > 1.0, 0.0, alpha_t_prime)
        alpha_t_prime = torch.where(x <= 0.0, 0.0, alpha_t_prime)
        return alpha_t_prime.reshape(batch_size, -1).to(original_precision)

    def __call__(self, t):
        move_chance = self.total_noise(t)
        alpha_t_prime = self.rate_noise(t)
        return 1 - move_chance, alpha_t_prime

    def compute_first_hitting_times(
        self, batch_size, length, device, dtype=torch.float64
    ):
        E = torch.empty(batch_size, length, dtype=dtype, device=device).exponential_()
        loc = self.loc.to(device)[:, -length:]
        factor = -torch.expm1(-E / self.k)
        T = loc[None, :] + self.b * factor
        return T.reshape(batch_size, -1)

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
            is_beginning = (to_permute.cumsum(-1) == 0) & (~to_permute[:, :, :1])
            is_end = (to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (
                ~to_permute[:, :, -1:]
            )

            ranking = torch.where(is_beginning, float("inf"), ranking)
            ranking = torch.where(is_end, float("-inf"), ranking)
            perm_indices = torch.argsort(
                ranking.cpu(), dim=-1, descending=True, stable=True
            ).to(device)
            if masked_tokens is not None:
                masked_tokens = masked_tokens.reshape(-1, block_size)
                masked_tokens |= is_end
                perm_flat = perm_indices.reshape(-1, block_size)
                for b in range(masked_tokens.shape[0]):
                    masked_indices = masked_tokens[b].nonzero(as_tuple=True)[0]
                    masked_assign = torch.isin(perm_flat[b], masked_indices)
                    perm_flat[b] = torch.cat(
                        [
                            perm_flat[b][~masked_assign],
                            perm_flat[b][masked_assign],
                        ],
                        dim=-1,
                    )
                perm_indices = perm_flat.reshape(batch_size, num_blocks, block_size)
            perm_indices = perm_indices.reshape(batch_size, num_blocks * block_size)
            return perm_indices
        num_total_blocks = t.shape[-1] // block_size * t.shape[0]
        to_permute = to_permute.reshape(-1, block_size)
        is_beginning = (to_permute.cumsum(-1) == 0) & (~to_permute[:, :1])
        is_end = (to_permute.flip(-1).cumsum(-1) == 0).flip(-1) & (~to_permute[:, -1:])

        T = self.compute_first_hitting_times(num_total_blocks, block_size, device)

        T = torch.where(is_beginning, float("inf"), T)
        T = torch.where(is_end, -float("inf"), T)
        perms = torch.argsort(T, dim=-1, stable=True, descending=True)
        if masked_tokens is not None:
            masked_tokens = masked_tokens.reshape(-1, block_size)
            masked_tokens |= is_end
            for b in range(masked_tokens.shape[0]):
                masked_indices = masked_tokens[b].nonzero(as_tuple=True)[0]
                masked_assign = torch.isin(perms[b], masked_indices)
                perms[b] = torch.cat(
                    [perms[b][~masked_assign], perms[b][masked_assign]], dim=-1
                )
        perms = perms.reshape(batch_size, -1, block_size)
        perms += torch.arange(num_blocks, device=device)[None, :, None] * block_size
        return perms.reshape(batch_size, -1)

    def compute_window_size(self):
        return self.block_size
