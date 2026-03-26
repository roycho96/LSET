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


if __name__ == "__main__":
    test_last_token_packed_matches_padded()
    test_mean_packed_matches_padded()
    test_cls_packed_matches_padded()
    print("All packed pooling tests passed!")
