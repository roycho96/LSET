"""Test BiEncoderTask forward pass."""

import torch

from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder
from lset.tasks.bi_encoder import BiEncoderTask


def test_bi_encoder_forward():
    """BiEncoderTask produces a scalar loss."""
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
    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    B, S = 4, 8
    query_batch = {
        "input_ids": torch.randint(0, 1000, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }
    doc_batch = {
        "input_ids": torch.randint(0, 1000, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }

    out = task(model, query_batch, doc_batch)
    assert "loss" in out
    assert out["loss"].dim() == 0  # scalar
    assert out["loss"].requires_grad
    assert "query_embeds" in out
    assert "doc_embeds" in out
    assert out["query_embeds"].shape == (B, 64)
    print(f"BiEncoder loss: {out['loss'].item():.4f}")


def test_bi_encoder_backward():
    """Loss backpropagates to model parameters."""
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
    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    B, S = 4, 8
    query_batch = {
        "input_ids": torch.randint(0, 1000, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }
    doc_batch = {
        "input_ids": torch.randint(0, 1000, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }

    out = task(model, query_batch, doc_batch)
    out["loss"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed to model parameters"


if __name__ == "__main__":
    test_bi_encoder_forward()
    test_bi_encoder_backward()
    print("All bi-encoder tests passed!")
