"""Smoke test for Qwen3Decoder forward pass."""

import torch

from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder


def test_import():
    """Validation: import works."""
    from lset.models.decoder.qwen3.model import Qwen3Decoder  # noqa: F811

    assert Qwen3Decoder is not None


def test_forward_shape():
    """Smoke test: random input produces correct output shape."""
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        max_position_embeddings=256,
    )
    model = Qwen3Decoder(config).to(dtype=torch.float32)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)

    out = model(input_ids, mask)
    assert "hidden_states" in out
    assert out["hidden_states"].shape == (B, S, 64)


def test_forward_with_lm_logits():
    """Smoke test: lm_logits returned when requested."""
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        max_position_embeddings=256,
    )
    model = Qwen3Decoder(config).to(dtype=torch.float32)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)

    out = model(input_ids, mask, return_lm_logits=True)
    assert "lm_logits" in out
    assert out["lm_logits"].shape == (B, S, 1000)


def test_forward_no_mask():
    """Forward pass without attention mask (pure causal)."""
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        max_position_embeddings=256,
    )
    model = Qwen3Decoder(config).to(dtype=torch.float32)
    input_ids = torch.randint(0, 1000, (2, 16))
    out = model(input_ids)
    assert out["hidden_states"].shape == (2, 16, 64)


if __name__ == "__main__":
    test_import()
    test_forward_shape()
    test_forward_with_lm_logits()
    test_forward_no_mask()
    print("All model smoke tests passed!")
