"""MiniLM text encoder with LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from .registry import TEXT_ENCODER_REGISTRY


@dataclass
class TextEncoderConfig:
    """Configuration for the textual encoder."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value")
    max_length: int = 256


class MiniLMTextEncoder(torch.nn.Module):
    """MiniLM encoder with parameter-efficient fine-tuning."""

    def __init__(self, config: Optional[TextEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or TextEncoderConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        base_model = AutoModel.from_pretrained(self.config.model_name)
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(base_model, peft_config)

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = self.model(**tokenized)
        return outputs.last_hidden_state


@TEXT_ENCODER_REGISTRY.register("minilm_l6_lora")
def build_minilm_text_encoder(**kwargs: Any) -> MiniLMTextEncoder:
    """Builder for the default MiniLM text encoder with LoRA adapters."""

    config = kwargs.pop("config", None)
    if isinstance(config, dict):
        kwargs = {**config, **kwargs}
        config = None
    if config is None:
        config = TextEncoderConfig(**kwargs)
    elif isinstance(config, TextEncoderConfig):
        if kwargs:
            raise ValueError(
                "Cannot supply extra kwargs when passing a TextEncoderConfig instance"
            )
    else:
        raise TypeError("config must be a TextEncoderConfig, dict, or None")
    return MiniLMTextEncoder(config)


__all__ = ["MiniLMTextEncoder", "TextEncoderConfig", "build_minilm_text_encoder"]
