"""BERT / XLM-RoBERTa config — covers both architectures."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    pad_token_id: int = 0
    # XLM-RoBERTa uses position offset (pos_ids start at padding_idx + 1)
    position_offset: int = 0

    @classmethod
    def from_hf_json(cls, path: str) -> BertConfig:
        with open(path) as f:
            data = json.load(f)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}

        # XLM-RoBERTa: position_ids start at padding_idx + 1
        model_type = data.get("model_type", "")
        if model_type in ("xlm-roberta", "roberta"):
            filtered["position_offset"] = data.get("pad_token_id", 1) + 1

        return cls(**filtered)
