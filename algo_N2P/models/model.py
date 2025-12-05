"""Unified multimodal forecasting model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .registry import (
    DECODER_REGISTRY,
    FUSION_REGISTRY,
    TEXT_ENCODER_REGISTRY,
    TIMESERIES_ENCODER_REGISTRY,
)


@dataclass
class ComponentSpec:
    """Declarative specification for a registry-backed component."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def update(self, data: Any) -> None:
        """Update the spec in-place from user-provided data."""

        if isinstance(data, str):
            self.name = data
            return
        if not isinstance(data, dict):
            raise TypeError(
                "ComponentSpec update expects a mapping or component name string"
            )
        if "name" in data and data["name"]:
            self.name = data["name"]
        params = data.get("params")
        if params:
            if not isinstance(params, dict):
                raise TypeError("'params' must be a mapping of keyword arguments")
            self.params.update(params)
        # Allow shorthand of placing kwargs at the top level.
        for key, value in data.items():
            if key not in {"name", "params"}:
                self.params[key] = value


@dataclass
class ModelConfig:
    """Configuration for the full multimodal model."""

    timeseries_encoder: ComponentSpec = field(
        default_factory=lambda: ComponentSpec("patchtst")
    )
    text_encoder: ComponentSpec = field(
        default_factory=lambda: ComponentSpec("minilm_l6_lora")
    )
    fusion: ComponentSpec = field(
        default_factory=lambda: ComponentSpec("cross_modal_qformer")
    )
    decoder: ComponentSpec = field(
        default_factory=lambda: ComponentSpec("diffusion_moe")
    )


class MultimodalPriceForecaster(torch.nn.Module):
    """Main model wiring together all modality-specific components."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.ts_encoder = TIMESERIES_ENCODER_REGISTRY.build(
            self.config.timeseries_encoder.name,
            **self.config.timeseries_encoder.params,
        )
        self.text_encoder = TEXT_ENCODER_REGISTRY.build(
            self.config.text_encoder.name,
            **self.config.text_encoder.params,
        )
        self.fusion = FUSION_REGISTRY.build(
            self.config.fusion.name,
            **self.config.fusion.params,
        )
        self.diffusion = DECODER_REGISTRY.build(
            self.config.decoder.name,
            **self.config.decoder.params,
        )

    def forward(
        self,
        timeseries_history: torch.Tensor,
        texts: list[str],
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        ts_tokens = self.ts_encoder(timeseries_history)
        text_tokens = self.text_encoder(texts)
        fused_context = self.fusion(ts_tokens, text_tokens)
        preds = self.diffusion(fused_context, noisy_future, timesteps)
        return preds


__all__ = [
    "MultimodalPriceForecaster",
    "ModelConfig",
    "ComponentSpec",
]
