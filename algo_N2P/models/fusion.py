"""Cross-modal Q-Former for late interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .registry import FUSION_REGISTRY


@dataclass
class QFormerConfig:
    """Configuration for the cross-modal Q-Former."""

    num_queries: int = 8
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 12
    dropout: float = 0.1


class CrossModalQFormer(nn.Module):
    """Query transformer attending to time-series and textual latents."""

    def __init__(self, config: Optional[QFormerConfig] = None) -> None:
        super().__init__()
        self.config = config or QFormerConfig()
        self.query_tokens = nn.Parameter(torch.randn(1, self.config.num_queries, self.config.d_model))
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.layers = nn.TransformerDecoder(encoder_layer, num_layers=self.config.num_layers)
        self.ts_proj = nn.Linear(self.config.d_model, self.config.d_model)
        self.text_proj = nn.Linear(self.config.d_model, self.config.d_model)

    def forward(self, ts_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse modalities via learned queries."""

        if ts_tokens.size(-1) != self.config.d_model:
            ts_tokens = self.ts_proj(ts_tokens)
        if text_tokens.size(-1) != self.config.d_model:
            text_tokens = self.text_proj(text_tokens)
        memory = torch.cat([ts_tokens, text_tokens], dim=1)
        query_tokens = self.query_tokens.expand(memory.size(0), -1, -1)
        fused = self.layers(query_tokens, memory)
        return fused


@FUSION_REGISTRY.register("cross_modal_qformer")
def build_cross_modal_qformer(**kwargs: Any) -> CrossModalQFormer:
    """Builder for the default cross-modal Q-Former fusion module."""

    config = kwargs.pop("config", None)
    if isinstance(config, dict):
        kwargs = {**config, **kwargs}
        config = None
    if config is None:
        config = QFormerConfig(**kwargs)
    elif isinstance(config, QFormerConfig):
        if kwargs:
            raise ValueError(
                "Cannot supply extra kwargs when passing a QFormerConfig instance"
            )
    else:
        raise TypeError("config must be a QFormerConfig, dict, or None")
    return CrossModalQFormer(config)


__all__ = ["CrossModalQFormer", "QFormerConfig", "build_cross_modal_qformer"]
