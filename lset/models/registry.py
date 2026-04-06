from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelSpec:
    config_cls: type
    model_cls: type
    weight_converter: Callable
    tp_plan_fn: Callable | None = None
    default_pooling: str = "last_token"
    default_padding_side: str = "left"
    post_pooling_fn: Callable | None = None


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


def detect_model_type(model_path: str) -> str:
    """Auto-detect model type from config.json."""
    import json
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])

    # Map HF model_type → LSET registry name
    type_map = {
        "qwen3": "qwen3",
        "llama": "llama",
        "llama_bidirec": "llama",
        "bert": "bert",
        "xlm-roberta": "xlm-roberta",
        "roberta": "xlm-roberta",
        "gemma3_text": "embeddinggemma",
    }

    if model_type in type_map:
        return type_map[model_type]

    # Fallback: check architectures list
    for arch in architectures:
        arch_lower = arch.lower()
        if "qwen3" in arch_lower:
            return "qwen3"
        if "llama" in arch_lower:
            return "llama"
        if "bert" in arch_lower and "roberta" not in arch_lower:
            return "bert"
        if "roberta" in arch_lower:
            return "xlm-roberta"
        if "gemma" in arch_lower:
            return "embeddinggemma"

    raise ValueError(
        f"Cannot auto-detect model type from {model_path}/config.json. "
        f"model_type={model_type!r}, architectures={architectures}. "
        f"Available types: {list(_REGISTRY.keys())}"
    )
