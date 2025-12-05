"""Diffusion decoder with sparse Mixture-of-Experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .registry import DECODER_REGISTRY


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion decoder."""

    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 12
    num_experts: int = 4
    top_k: int = 2
    dropout: float = 0.1
    forecast_steps: int = 30
    price_channels: int = 5


class SparseMoE(nn.Module):
    """Simple sparse MoE layer with top-k gating."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        logits = self.gate(x_flat)
        weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = topk_indices[:, k]
            expert_weight = topk_weights[:, k].unsqueeze(-1)
            expert_out = torch.zeros_like(x_flat)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                if mask.any():
                    expert_out[mask] = expert(x_flat[mask])
            output += expert_out * expert_weight
        return output.view(batch, seq_len, hidden)


class DiffusionDecoder(nn.Module):
    """Conditional diffusion decoder for future price trajectories."""

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList([nn.TransformerEncoder(decoder_layer, num_layers=1) for _ in range(self.config.num_layers)])
        self.moe = SparseMoE(self.config.d_model, self.config.num_experts, self.config.top_k)
        self.input_proj = nn.Linear(self.config.price_channels, self.config.d_model)
        self.proj_out = nn.Linear(self.config.d_model, self.config.price_channels)

    def forward(self, context: torch.Tensor, noisy_future: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Run diffusion denoising conditioned on fused context."""

        hidden = self.input_proj(noisy_future)
        time_emb = self.time_embedding(timesteps.float().unsqueeze(-1)).unsqueeze(1)
        hidden = hidden + time_emb
        for layer in self.layers:
            hidden = layer(hidden)
            hidden = hidden + self.moe(hidden)
        fused_context = context.mean(dim=1, keepdim=True)
        hidden = hidden + fused_context
        return self.proj_out(hidden)


@DECODER_REGISTRY.register("diffusion_moe")
def build_diffusion_decoder(**kwargs: Any) -> DiffusionDecoder:
    """Builder for the sparse-MoE diffusion decoder."""

    config = kwargs.pop("config", None)
    if isinstance(config, dict):
        kwargs = {**config, **kwargs}
        config = None
    if config is None:
        config = DiffusionConfig(**kwargs)
    elif isinstance(config, DiffusionConfig):
        if kwargs:
            raise ValueError(
                "Cannot supply extra kwargs when passing a DiffusionConfig instance"
            )
    else:
        raise TypeError("config must be a DiffusionConfig, dict, or None")
    return DiffusionDecoder(config)


__all__ = ["DiffusionDecoder", "DiffusionConfig", "SparseMoE", "build_diffusion_decoder"]
