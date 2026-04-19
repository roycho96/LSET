"""Tests for pooling strategies."""

import torch

from lset.models.pooling import pool


def test_last_token_pooling():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    mask = torch.tensor([[1, 1, 1]])
    emb = pool(hidden, mask, "last_token", normalize=False)
    assert emb.shape == (1, 2)
    assert torch.allclose(emb, torch.tensor([[5.0, 6.0]]))


def test_mean_pooling():
    hidden = torch.tensor([[[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]])
    mask = torch.tensor([[0, 1, 1]])  # first token is padding
    emb = pool(hidden, mask, "mean", normalize=False)
    assert emb.shape == (1, 2)
    assert torch.allclose(emb, torch.tensor([[3.0, 6.0]]))


def test_cls_pooling():
    hidden = torch.tensor([[[10.0, 20.0], [1.0, 2.0]]])
    mask = torch.tensor([[1, 1]])
    emb = pool(hidden, mask, "cls", normalize=False)
    assert torch.allclose(emb, torch.tensor([[10.0, 20.0]]))


def test_normalize():
    hidden = torch.tensor([[[3.0, 4.0]]])
    mask = torch.tensor([[1]])
    emb = pool(hidden, mask, "last_token", normalize=True)
    assert torch.allclose(emb.norm(dim=-1), torch.tensor([1.0]), atol=1e-6)


def test_weighted_mean():
    hidden = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    mask = torch.tensor([[0, 1, 1]])  # padding, then 2 real tokens
    emb = pool(hidden, mask, "weighted_mean", normalize=False)
    assert emb.shape == (1, 2)
    # weights for real tokens: pos1=1, pos2=2 → normalized: 1/3, 2/3
    expected = (1.0 / 3) * torch.tensor([1.0, 1.0]) + (2.0 / 3) * torch.tensor([2.0, 2.0])
    assert torch.allclose(emb.squeeze(0), expected, atol=1e-5)


if __name__ == "__main__":
    test_last_token_pooling()
    test_mean_pooling()
    test_cls_pooling()
    test_normalize()
    test_weighted_mean()
    print("All pooling tests passed!")
