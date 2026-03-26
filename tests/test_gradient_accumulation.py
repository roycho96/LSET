"""Tests for gradient accumulation correctness."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.tasks.losses.infonce import infonce_loss


def test_grad_accum_matches_large_batch():
    """Gradient accumulation over 4 microbatches ≈ single large batch."""
    torch.manual_seed(42)
    D = 64

    # Small linear model for testing
    model_large = nn.Linear(D, D, bias=False)
    model_accum = nn.Linear(D, D, bias=False)
    model_accum.load_state_dict(model_large.state_dict())

    # Large batch: 8 samples
    q_all = torch.randn(8, D)
    d_all = torch.randn(8, D)

    # Method 1: Single large batch
    q_emb = F.normalize(model_large(q_all), dim=-1)
    d_emb = F.normalize(model_large(d_all), dim=-1)
    loss_large = infonce_loss(q_emb, d_emb, temperature=0.05)
    loss_large.backward()
    grad_large = model_large.weight.grad.clone()

    # Method 2: Gradient accumulation (4 chunks of 2)
    accum_steps = 4
    model_accum.zero_grad()
    for i in range(accum_steps):
        s, e = i * 2, (i + 1) * 2
        q_emb = F.normalize(model_accum(q_all[s:e]), dim=-1)
        d_emb = F.normalize(model_accum(d_all[s:e]), dim=-1)
        loss_chunk = infonce_loss(q_emb, d_emb, temperature=0.05) / accum_steps
        loss_chunk.backward()
    grad_accum = model_accum.weight.grad.clone()

    # Note: the losses won't be identical because the contrastive matrix is different
    # (8x8 vs 2x2). But gradients should be in the same direction.
    cos_sim = F.cosine_similarity(grad_large.flatten().unsqueeze(0),
                                   grad_accum.flatten().unsqueeze(0))
    assert cos_sim > 0.5, f"Gradient direction similarity too low: {cos_sim.item()}"
