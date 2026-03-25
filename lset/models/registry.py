from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ModelSpec:
    config_cls: type
    model_cls: type
    weight_converter: Callable
    tp_plan_fn: Callable | None = None
    default_pooling: str = "last_token"
    default_padding_side: str = "left"


_REGISTRY: dict[str, ModelSpec] = {}
_ALIASES: dict[str, str] = {}


def register_model(name: str, spec: ModelSpec) -> None:
    _REGISTRY[name] = spec


def register_alias(alias: str, target: str) -> None:
    _ALIASES[alias] = target


def get_model_spec(name: str) -> ModelSpec:
    resolved = _ALIASES.get(name, name)
    if resolved not in _REGISTRY:
        raise KeyError(f"Unknown model: {name!r} (resolved to {resolved!r}). "
                       f"Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[resolved]
