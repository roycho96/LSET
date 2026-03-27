from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    # Llama3 RoPE scaling
    rope_scaling: dict | None = None

    @classmethod
    def from_hf_json(cls, path: str) -> LlamaConfig:
        with open(path) as f:
            data = json.load(f)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
