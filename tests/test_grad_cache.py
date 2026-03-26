"""Tests for GradCache — verify loss matches standard forward."""

import pytest
import torch
from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.grad_cache import GradCacheWrapper


def _make_model_and_data(B=4, S=8):
    """Create small model and synthetic batches."""
    config = Qwen3Config(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=100, max_position_embeddings=64)
    model = Qwen3Decoder(config).to(dtype=torch.float32)

    query_batch = {
        "input_ids": torch.randint(0, 100, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }
    doc_batch = {
        "input_ids": torch.randint(0, 100, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
    }
    return config, model, query_batch, doc_batch


def test_grad_cache_full_chunk_matches_standard():
    """chunk_size=B should give same loss as standard forward."""
    torch.manual_seed(42)
    _, model, query_batch, doc_batch = _make_model_and_data(B=4)

    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    # Standard loss
    model_std = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_std.load_state_dict(model.state_dict())
    out = task(model_std, query_batch, doc_batch)
    std_loss = out["loss"].item()

    # GradCache with chunk_size = full batch
    model_gc = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_gc.load_state_dict(model.state_dict())
    gc = GradCacheWrapper(task, chunk_size=4)
    gc_loss = gc(model_gc, query_batch, doc_batch).item()

    print(f"Standard loss: {std_loss:.6f}, GradCache loss: {gc_loss:.6f}")
    assert abs(std_loss - gc_loss) < 1e-5, f"Loss mismatch: {std_loss} vs {gc_loss}"


def test_grad_cache_chunk1_matches_standard():
    """chunk_size=1 should give same loss as standard."""
    torch.manual_seed(123)
    _, model, query_batch, doc_batch = _make_model_and_data(B=4)

    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    # Standard
    model_std = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_std.load_state_dict(model.state_dict())
    out = task(model_std, query_batch, doc_batch)
    std_loss = out["loss"].item()

    # GradCache chunk=1
    model_gc = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_gc.load_state_dict(model.state_dict())
    gc = GradCacheWrapper(task, chunk_size=1)
    gc_loss = gc(model_gc, query_batch, doc_batch).item()

    print(f"Standard loss: {std_loss:.6f}, GradCache(chunk=1) loss: {gc_loss:.6f}")
    assert abs(std_loss - gc_loss) < 1e-5, f"Loss mismatch: {std_loss} vs {gc_loss}"


def test_grad_cache_grads_match():
    """GradCache gradients should match standard gradients."""
    torch.manual_seed(99)
    _, model, query_batch, doc_batch = _make_model_and_data(B=4)

    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    # Standard grads
    model_std = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_std.load_state_dict(model.state_dict())
    out = task(model_std, query_batch, doc_batch)
    out["loss"].backward()
    std_grads = {n: p.grad.clone() for n, p in model_std.named_parameters() if p.grad is not None}

    # GradCache grads
    model_gc = Qwen3Decoder(model.config).to(dtype=torch.float32)
    model_gc.load_state_dict(model.state_dict())
    gc = GradCacheWrapper(task, chunk_size=2)
    gc(model_gc, query_batch, doc_batch)
    gc_grads = {n: p.grad.clone() for n, p in model_gc.named_parameters() if p.grad is not None}

    assert set(std_grads.keys()) == set(gc_grads.keys()), "Gradient key mismatch"

    max_diff = 0.0
    for name in std_grads:
        diff = (std_grads[name] - gc_grads[name]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"Max gradient difference: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Gradient mismatch: max diff {max_diff}"


if __name__ == "__main__":
    test_grad_cache_full_chunk_matches_standard()
    test_grad_cache_chunk1_matches_standard()
    test_grad_cache_grads_match()
    print("All GradCache tests passed!")
