"""PatchTST encoder for financial time-series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .registry import TIMESERIES_ENCODER_REGISTRY


@dataclass
class PatchTSTConfig:
    """Configuration parameters for the PatchTST encoder."""

    input_channels: int = 5
    patch_length: int = 6
    stride: int = 3
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.1


class PatchEmbedding(nn.Module):
    """Patchify time-series and project to the model dimension."""

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.unfold = nn.Unfold(kernel_size=(config.patch_length, 1), stride=(config.stride, 1))
        patch_dim = config.input_channels * config.patch_length
        self.proj = nn.Linear(patch_dim, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        x = x.transpose(1, 2).unsqueeze(-1)  # (B, C, L, 1)
        patches = self.unfold(x).transpose(1, 2)
        return self.proj(patches)


class PatchTSTEncoder(nn.Module):
    """Transformer encoder operating on time-series patches."""

    def __init__(self, config: Optional[PatchTSTConfig] = None) -> None:
        super().__init__()
        self.config = config or PatchTSTConfig()
        self.embedding = PatchEmbedding(self.config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, self.config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embedding(x)
        max_len = self.positional_encoding.size(1)
        if patches.size(1) > max_len:
            raise ValueError("Patch sequence longer than positional embedding")
        patches = patches + self.positional_encoding[:, : patches.size(1), :]
        return self.encoder(patches)


@TIMESERIES_ENCODER_REGISTRY.register("patchtst")
def build_patchtst_encoder(**kwargs: Any) -> PatchTSTEncoder:
    """Builder that adapts dict-style kwargs into :class:`PatchTSTEncoder`."""

    config = kwargs.pop("config", None)
    if isinstance(config, dict):
        kwargs = {**config, **kwargs}
        config = None
    if config is None:
        config = PatchTSTConfig(**kwargs)
    elif isinstance(config, PatchTSTConfig):
        if kwargs:
            raise ValueError(
                "Cannot provide additional kwargs when passing a PatchTSTConfig instance"
            )
    else:
        raise TypeError("config must be a PatchTSTConfig, dict, or None")
    return PatchTSTEncoder(config)


__all__ = ["PatchTSTEncoder", "PatchTSTConfig", "build_patchtst_encoder"]
