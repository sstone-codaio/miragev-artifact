from __future__ import annotations

from typing import Any, Dict

from .base import ModelAdapter


_REGISTRY: Dict[str, Any] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def create_adapter(name: str, **kwargs) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. Available: {sorted(_REGISTRY.keys())}. "
            f"Add your adapter in miragev/models and register it."
        )
    return _REGISTRY[name](**kwargs)
