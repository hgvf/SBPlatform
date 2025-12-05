"""Registries for model components to enable easy swapping."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class ComponentRegistry:
    """Simple name-to-builder registry for interchangeable modules."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a builder under ``name``.

        The decorated callable should return an instantiated module when invoked
        with keyword arguments describing its configuration.
        """

        def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._builders:
                raise ValueError(f"{self.kind} builder '{name}' already registered")
            self._builders[name] = builder
            return builder

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate the builder registered under ``name``."""

        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available options: {available}."
            )
        return self._builders[name](**kwargs)

    def get(self, name: str) -> Callable[..., Any]:
        """Return the underlying builder callable."""

        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available options: {available}."
            )
        return self._builders[name]

    def available(self) -> Tuple[str, ...]:
        """Return a sorted tuple of registered builder names."""

        return tuple(sorted(self._builders))


TIMESERIES_ENCODER_REGISTRY = ComponentRegistry("time-series encoder")
TEXT_ENCODER_REGISTRY = ComponentRegistry("text encoder")
FUSION_REGISTRY = ComponentRegistry("fusion module")
DECODER_REGISTRY = ComponentRegistry("decoder")


__all__ = [
    "ComponentRegistry",
    "TIMESERIES_ENCODER_REGISTRY",
    "TEXT_ENCODER_REGISTRY",
    "FUSION_REGISTRY",
    "DECODER_REGISTRY",
]
