"""Tests for packed pooling — compare with padded pooling."""

import torch
from lset.tasks.pooling import pool
from lset.tasks.packed_pooling import packed_pool


def test_last_token_packed_matches_padded():
    """last_token packed pooling matches padded pooling."""
    # Two sequences: [a, b, c] and [d, e]
    h1 = torch.randn(3, 8)
    h2 = torch.randn(2, 8)

    # Packed
    packed_hidden = torch.cat([h1, h2], dim=0)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    packed_emb = packed_pool(packed_hidden, cu_seqlens, "last_token", normalize=False)

    # Padded (left-pad h2 to length 3)
    padded_hidden = torch.stack([
        h1,
        torch.cat([torch.zeros(1, 8), h2], dim=0),
    ])
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    padded_emb = pool(padded_hidden, mask, "last_token", normalize=False)

    assert torch.allclose(packed_emb, padded_emb, atol=1e-6)


def test_mean_packed_matches_padded():
    """mean packed pooling matches padded pooling."""
    h1 = torch.randn(3, 8)
    h2 = torch.randn(2, 8)

    # Packed
    packed_hidden = torch.cat([h1, h2], dim=0)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    packed_emb = packed_pool(packed_hidden, cu_seqlens, "mean", normalize=False)

    # Padded
    padded_hidden = torch.stack([
        h1,
        torch.cat([torch.zeros(1, 8), h2], dim=0),
    ])
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    padded_emb = pool(padded_hidden, mask, "mean", normalize=False)

    assert torch.allclose(packed_emb, padded_emb, atol=1e-6)


def test_cls_packed_matches_padded():
    """cls packed pooling matches padded pooling (first real token)."""
    h1 = torch.randn(3, 8)
    h2 = torch.randn(2, 8)

    # Packed: first token of each seq
    packed_hidden = torch.cat([h1, h2], dim=0)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    packed_emb = packed_pool(packed_hidden, cu_seqlens, "cls", normalize=False)

    assert torch.allclose(packed_emb[0], h1[0], atol=1e-6)
    assert torch.allclose(packed_emb[1], h2[0], atol=1e-6)


def test_mean_packed_single_sequence():
    """Edge case: single sequence."""
    h = torch.randn(5, 8)
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    emb = packed_pool(h, cu_seqlens, "mean", normalize=False)
    expected = h.mean(0, keepdim=True)
    assert torch.allclose(emb, expected, atol=1e-6)


def test_mean_packed_length_one():
    """Edge case: sequences of length 1."""
    h = torch.randn(3, 8)
    cu_seqlens = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    emb = packed_pool(h, cu_seqlens, "mean", normalize=False)
    assert torch.allclose(emb, h, atol=1e-6)


def test_mean_packed_gradient_flows():
    """Gradient flows through scatter_add_ mean pooling."""
    h = torch.randn(5, 8, requires_grad=True)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    emb = packed_pool(h, cu_seqlens, "mean", normalize=False)
    emb.sum().backward()
    assert h.grad is not None
    assert h.grad.shape == h.shape


if __name__ == "__main__":
    test_last_token_packed_matches_padded()
    test_mean_packed_matches_padded()
    test_cls_packed_matches_padded()
    test_mean_packed_single_sequence()
    test_mean_packed_length_one()
    test_mean_packed_gradient_flows()
    print("All packed pooling tests passed!")
