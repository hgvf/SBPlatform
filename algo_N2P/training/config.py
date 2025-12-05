"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import yaml

from ..data.dataset import WindowConfig
from ..models.model import ComponentSpec, ModelConfig


@dataclass
class OptimConfig:
    """Optimiser hyper-parameters."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.05
    max_steps: int = 100_000


@dataclass
class TrainingConfig:
    """Top-level configuration for experiments."""

    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimiser: OptimConfig = field(default_factory=OptimConfig)
    batch_size: int = 16
    grad_clip: float = 1.0
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Construct a :class:`TrainingConfig` from a nested mapping."""

        config = cls()
        if data:
            _update_dataclass(config, data)
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load a :class:`TrainingConfig` from a YAML file."""

        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("Training configuration YAML must map keys to values")
        return cls.from_dict(loaded)


def _update_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    """Recursively merge ``updates`` into ``instance``."""

    if not isinstance(updates, Mapping):
        raise TypeError("Updates must be provided as a mapping")
    for field_info in fields(instance):
        if field_info.name not in updates:
            continue
        value = updates[field_info.name]
        current = getattr(instance, field_info.name)
        if value is None:
            continue
        if isinstance(current, ComponentSpec):
            current.update(value)
        elif is_dataclass(current):
            _update_dataclass(current, value)
        elif isinstance(current, dict) and isinstance(value, Mapping):
            merged = current.copy()
            merged.update(value)
            setattr(instance, field_info.name, merged)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = type(current)(value)
            setattr(instance, field_info.name, value)
    return instance


def load_training_config(path: str) -> TrainingConfig:
    """Convenience wrapper around :meth:`TrainingConfig.from_yaml`."""

    return TrainingConfig.from_yaml(path)


__all__ = [
    "TrainingConfig",
    "OptimConfig",
    "_update_dataclass",
    "load_training_config",
]
