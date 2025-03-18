from abc import ABC, abstractmethod

import torch


class Noise(ABC):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def __call__(
        self, t: torch.Tensor | float
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

    @abstractmethod
    def rate_noise(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """
        Noise rate of change, i.e., g(t)
        """
        pass

    @abstractmethod
    def total_noise(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """
        Total noise ie \\int_0^t g(t) dt + g(0)
        """
        pass


class CosineNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        cos = torch.cos(t * torch.pi / 2)
        return -torch.log(self.eps + (1 - self.eps) * cos)


class CosineSqrNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        cos = (1 - self.eps) * (torch.cos(t * torch.pi / 2) ** 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        cos = torch.cos(t * torch.pi / 2) ** 2
        return -torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
    def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
        super().__init__()
        self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
        self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

    def rate_noise(self, t):
        return self.sigma_max - self.sigma_min

    def total_noise(self, t):
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        return (sigma_t - self.sigma_min) / (self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
    def __init__(self, sigma_min=1e-3, sigma_max=1):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t):
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        t = -torch.expm1(-sigma_t) / (1 - self.eps)
        return t
