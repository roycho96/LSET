"""Tests for Llama decoder model."""

import torch

from lset.models.decoder.llama.config import LlamaConfig
from lset.models.decoder.llama.model import LlamaDecoder


def test_config_from_hf_json():
    config = LlamaConfig.from_hf_json("/home/roy/models/llama-nemotron-embed-1b-v2/config.json")
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 16
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 64
    assert config.intermediate_size == 8192
    assert config.vocab_size == 128256
    assert config.rope_theta == 500_000.0


def test_forward_shape():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
    )
    model = LlamaDecoder(config)
    x = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    out = model(x, mask)
    assert out["hidden_states"].shape == (2, 16, 128)


def test_bidirectional_attention():
    """Llama-Nemotron uses bidirectional attention — no causal mask."""
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
    )
    model = LlamaDecoder(config)
    # Without mask, should NOT apply causal masking
    x = torch.randint(0, 1000, (1, 8))
    out = model(x)
    assert out["hidden_states"].shape == (1, 8, 64)
    assert not torch.isnan(out["hidden_states"]).any()
