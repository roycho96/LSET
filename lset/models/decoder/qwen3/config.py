from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Qwen3Config:
    vocab_size: int = 151669
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    @classmethod
    def from_hf_json(cls, path: str) -> Qwen3Config:
        with open(path) as f:
            data = json.load(f)
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
