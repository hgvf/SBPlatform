"""Utilities for forward and reverse diffusion schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiffusionSchedule:
    """Simple linear beta diffusion schedule."""

    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)

        self._betas = betas
        self._alphas = alphas
        self._alpha_cumprod = alpha_cumprod
        self._alpha_cumprod_prev = alpha_cumprod_prev
        self._sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self._sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        self._sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.device = torch.device("cpu")

    @property
    def betas(self) -> Tensor:
        return self._betas

    @property
    def alphas(self) -> Tensor:
        return self._alphas

    @property
    def alpha_cumprod(self) -> Tensor:
        return self._alpha_cumprod

    @property
    def alpha_cumprod_prev(self) -> Tensor:
        return self._alpha_cumprod_prev

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        """Move schedule tensors to ``device``."""

        device = torch.device(device)
        self._betas = self._betas.to(device)
        self._alphas = self._alphas.to(device)
        self._alpha_cumprod = self._alpha_cumprod.to(device)
        self._alpha_cumprod_prev = self._alpha_cumprod_prev.to(device)
        self._sqrt_alpha_cumprod = self._sqrt_alpha_cumprod.to(device)
        self._sqrt_alpha_cumprod_prev = self._sqrt_alpha_cumprod_prev.to(device)
        self._sqrt_one_minus_alpha_cumprod = self._sqrt_one_minus_alpha_cumprod.to(device)
        self.device = device
        return self

    def _gather(self, values: Tensor, timesteps: Tensor) -> Tensor:
        gathered = values[timesteps]
        return gathered.view(timesteps.shape[0], *([1] * 2))

    def normalise_timesteps(self, timesteps: Tensor) -> Tensor:
        """Scale integer timesteps into the ``[0, 1]`` range for conditioning."""

        if self.num_steps <= 1:
            return torch.zeros_like(timesteps, dtype=torch.float32)
        return timesteps.float() / float(self.num_steps - 1)

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Diffuse ``sample`` forward to ``x_t`` using ``noise`` at ``timesteps``."""

        sqrt_alpha_cumprod = self._gather(self._sqrt_alpha_cumprod, timesteps)
        sqrt_one_minus = self._gather(self._sqrt_one_minus_alpha_cumprod, timesteps)
        return sqrt_alpha_cumprod * sample + sqrt_one_minus * noise

    def predict_start_from_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Compute the predicted ``x_0`` from noisy sample and noise estimate."""

        sqrt_alpha_cumprod = self._gather(self._sqrt_alpha_cumprod, timesteps)
        sqrt_one_minus = self._gather(self._sqrt_one_minus_alpha_cumprod, timesteps)
        return (sample - sqrt_one_minus * noise) / torch.clamp(sqrt_alpha_cumprod, min=1e-8)

    def step(self, model_output: Tensor, timesteps: Tensor, sample: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Perform one reverse diffusion step."""

        betas_t = self._gather(self._betas, timesteps)
        alphas_t = self._gather(self._alphas, timesteps)
        alpha_cumprod_t = self._gather(self._alpha_cumprod, timesteps)
        alpha_cumprod_prev = self._gather(self._alpha_cumprod_prev, timesteps)

        pred_original = self.predict_start_from_noise(sample, model_output, timesteps)

        # If t == 0 return the predicted original sample
        if torch.all(timesteps == 0):
            return pred_original

        sqrt_alpha_t = torch.sqrt(alphas_t)
        sqrt_alpha_cumprod_prev = self._gather(self._sqrt_alpha_cumprod_prev, timesteps)

        coeff1 = betas_t * sqrt_alpha_cumprod_prev / torch.clamp(1.0 - alpha_cumprod_t, min=1e-8)
        coeff2 = sqrt_alpha_t * (1.0 - alpha_cumprod_prev) / torch.clamp(1.0 - alpha_cumprod_t, min=1e-8)
        posterior_mean = coeff1 * pred_original + coeff2 * sample
        posterior_variance = betas_t * (1.0 - alpha_cumprod_prev) / torch.clamp(1.0 - alpha_cumprod_t, min=1e-8)
        noise = torch.randn_like(sample) if generator is None else torch.randn_like(sample, generator=generator)
        return posterior_mean + torch.sqrt(torch.clamp(posterior_variance, min=1e-8)) * noise


__all__ = ["DiffusionSchedule"]

