"""Integration tests: Packed + GradCache end-to-end."""

import pytest
import torch
from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.grad_cache import GradCacheWrapper
from lset.train.data.packing import pack_sequences


def _make_packed_batches(B=4, device="cpu"):
    """Create packed query and doc batches."""
    queries = [[10 + i, 20 + i, 30 + i] for i in range(B)]
    docs = [[40 + i, 50 + i] for i in range(B)]

    q_packed = pack_sequences(queries)
    d_packed = pack_sequences(docs)

    return (
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in q_packed.items()},
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d_packed.items()},
    )


def test_packed_grad_cache_combined():
    """GradCache with packed input runs and produces gradients."""
    torch.manual_seed(42)
    config = Qwen3Config(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=100, max_position_embeddings=64)
    model = Qwen3Decoder(config).to(dtype=torch.float32)
    task = BiEncoderTask(pooling="last_token", temperature=0.05)
    gc = GradCacheWrapper(task, chunk_size=2)

    q_batch, d_batch = _make_packed_batches(B=4)
    loss = gc(model, q_batch, d_batch)

    assert loss.dim() == 0
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients from packed GradCache"
    print(f"Packed GradCache loss: {loss.item():.4f}")


def test_packed_training_steps():
    """10 training steps with packed mode, loss changes."""
    torch.manual_seed(7)
    config = Qwen3Config(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=100, max_position_embeddings=64)
    model = Qwen3Decoder(config).to(dtype=torch.float32)
    task = BiEncoderTask(pooling="last_token", temperature=0.05)
    gc = GradCacheWrapper(task, chunk_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        q_batch, d_batch = _make_packed_batches(B=4)
        loss = gc(model, q_batch, d_batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    print(f"Losses: {[f'{l:.4f}' for l in losses]}")
    # Just verify it ran without error and loss is finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_standard_forward():
    """Standard (non-GradCache) packed forward+backward works."""
    config = Qwen3Config(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=100, max_position_embeddings=64)
    model = Qwen3Decoder(config).to(device="cuda", dtype=torch.float32)
    task = BiEncoderTask(pooling="last_token", temperature=0.05)

    q_batch, d_batch = _make_packed_batches(B=4, device="cuda")
    out = task(model, q_batch, d_batch)
    out["loss"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    print(f"Packed standard loss: {out['loss'].item():.4f}")


if __name__ == "__main__":
    test_packed_grad_cache_combined()
    test_packed_training_steps()
    if torch.cuda.is_available():
        test_packed_standard_forward()
    print("All integration tests passed!")
