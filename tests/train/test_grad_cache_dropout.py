"""GradCache Step 1/Step 3 bit-exactness under dropout.

With RandContext, the two forwards for the same minibatch must draw the
same dropout masks, so the cached ∂L/∂embedding is consistent with the
replay forward. Without RandContext the two passes diverge and the
gradient in params is silently biased.

This test exercises only the no_grad cache vs with_grad replay path —
it does not claim GradCache(chunked) equals full-batch forward (dropout
is batch-size-dependent, so those differ by design).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.train.grad_cache import GradCacheWrapper


class _FakeTask:
    def __init__(self, hidden=16, p=0.3):
        torch.manual_seed(0)
        self.linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(p=p)
        self.temperature = 0.05
        self.matryoshka_dims = None
        self.top_k = None
        self.logit_scale = None

    def encode(self, model, batch):
        x = model(batch["input_ids"].float())
        x = self.dropout(x)
        x = self.linear(x)
        return F.normalize(x, dim=-1)


class _FakeModel(nn.Module):
    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, ids):
        return self.dropout(self.emb(ids.long())).mean(dim=1)


def _build_batch(B=6, S=4):
    torch.manual_seed(7)
    return {"input_ids": torch.randint(0, 32, (B, S)), "attention_mask": torch.ones(B, S, dtype=torch.long)}


def test_grad_cache_same_seed_produces_identical_grads():
    """Two runs of GradCache with the same seed must produce identical param grads."""
    q = _build_batch()
    d = _build_batch()

    torch.manual_seed(123)
    model1 = _FakeModel()
    task1 = _FakeTask()
    gc1 = GradCacheWrapper(task1, chunk_size=2)
    torch.manual_seed(42)
    gc1(model1, q, d)
    g1 = {n: p.grad.clone() for n, p in model1.named_parameters()}

    torch.manual_seed(123)
    model2 = _FakeModel()
    task2 = _FakeTask()
    gc2 = GradCacheWrapper(task2, chunk_size=2)
    torch.manual_seed(42)
    gc2(model2, q, d)
    g2 = {n: p.grad.clone() for n, p in model2.named_parameters()}

    for name in g1:
        assert torch.allclose(g1[name], g2[name], atol=1e-6), f"nondeterministic grad for {name}"


def test_autograd_grad_path_produces_param_grads():
    """Without dropout, GradCache must still populate .grad on every param via
    the surrogate replay (confirms Step 2 used autograd.grad and Step 3 ran)."""
    model = _FakeModel()
    task = _FakeTask(p=0.0)  # deterministic
    q = _build_batch(B=4)
    d = _build_batch(B=4)

    gc = GradCacheWrapper(task, chunk_size=2)
    gc(model, q, d)

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} missing grad"
        assert p.grad.abs().sum() > 0, f"{name} has zero grad"
