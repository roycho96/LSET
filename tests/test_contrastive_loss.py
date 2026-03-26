"""Tests for label-matrix-aware contrastive loss."""

import torch
import torch.nn.functional as F
import pytest

from lset.tasks.losses.contrastive import contrastive_loss
from lset.tasks.losses.infonce import infonce_loss


def test_single_positive_matches_infonce():
    """Diagonal single-positive should match old infonce_loss."""
    torch.manual_seed(42)
    B, D = 8, 64
    q = F.normalize(torch.randn(B, D), dim=-1)
    d = F.normalize(torch.randn(B, D), dim=-1)
    temp = 0.05

    # InfoNCE
    loss_old = infonce_loss(q, d, temp)

    # Contrastive with diagonal labels
    labels = torch.zeros(B, B)
    labels.fill_diagonal_(1.0)
    loss_new = contrastive_loss(q, d, labels, temp)

    assert torch.allclose(loss_old, loss_new, atol=1e-5), \
        f"InfoNCE={loss_old.item()}, Contrastive={loss_new.item()}"


def test_multi_positive_produces_gradient():
    """Multi-positive contrastive loss should produce valid gradients."""
    torch.manual_seed(42)
    B, D = 4, 64
    q = torch.randn(B, D, requires_grad=True)
    q_norm = F.normalize(q, dim=-1)
    d = F.normalize(torch.randn(B, D), dim=-1)
    temp = 0.05

    # Multi positive: diagonal + one extra
    labels = torch.zeros(B, B)
    labels.fill_diagonal_(1.0)
    for i in range(B):
        labels[i, (i + 1) % B] = 1.0
    loss = contrastive_loss(q_norm, d, labels, temp)
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0
    assert loss.item() > 0


def test_soft_label_gradient():
    """Soft label loss should produce gradients."""
    B, D = 4, 64
    q_raw = torch.randn(B, D, requires_grad=True)
    q = F.normalize(q_raw, dim=-1)
    d = F.normalize(torch.randn(B, D), dim=-1)
    labels = torch.zeros(B, B)
    labels.fill_diagonal_(1.0)
    scores = torch.zeros(B, B)
    scores.fill_diagonal_(1.0)

    loss = contrastive_loss(q, d, labels, 0.05, scores=scores)
    loss.backward()
    assert q_raw.grad is not None
    assert q_raw.grad.abs().sum() > 0


def test_ignore_labels():
    """Label=-1 entries should not affect the loss."""
    torch.manual_seed(42)
    B, D = 4, 64
    q = F.normalize(torch.randn(B, D), dim=-1)
    d = F.normalize(torch.randn(B, D), dim=-1)
    temp = 0.05

    # All non-diagonal as in-batch negative (0)
    labels1 = torch.zeros(B, B)
    labels1.fill_diagonal_(1.0)
    loss1 = contrastive_loss(q, d, labels1, temp)

    # Mark some off-diagonal as ignore (-1)
    labels2 = labels1.clone()
    labels2[0, 1] = -1
    labels2[1, 0] = -1
    loss2 = contrastive_loss(q, d, labels2, temp)

    # They should differ since denominator changes
    assert not torch.allclose(loss1, loss2), "Ignore labels should affect the loss"
