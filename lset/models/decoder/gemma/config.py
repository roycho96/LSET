from __future__ import annotations

import json

from dataclasses import dataclass
from dataclasses import field


@dataclass
class GemmaConfig:
    vocab_size: int = 262144
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 24
    num_attention_heads: int = 3
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    sliding_window: int = 512
    query_pre_attn_scalar: int = 256
    # Layer types: "sliding_attention" or "full_attention"
    layer_types: list[str] = field(default_factory=list)

    @classmethod
    def from_hf_json(cls, path: str) -> GemmaConfig:
        with open(path) as f:
            data = json.load(f)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
