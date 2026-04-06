"""Tests for LR schedulers."""

import torch
from lset.train.scheduler import build_scheduler


def test_cosine_decreases():
    """Cosine scheduler should decrease LR over training."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, "cosine", max_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[0] > lrs[-1], "Cosine LR should decrease"
    assert lrs[-1] < 0.1, "Cosine LR should be near 0 at end"


def test_warmup_cosine():
    """Warmup + cosine should increase then decrease."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, "cosine", max_steps=100, warmup_steps=20)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Warmup: should increase
    assert lrs[10] > lrs[0], "LR should increase during warmup"
    # After warmup: should decrease
    assert lrs[50] > lrs[90], "LR should decrease after warmup"


def test_constant_scheduler():
    """Constant scheduler should maintain LR."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    scheduler = build_scheduler(optimizer, "constant", max_steps=50)

    for _ in range(50):
        optimizer.step()
        scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - 0.5) < 1e-6


def test_linear_scheduler():
    """Linear scheduler should decrease linearly."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, "linear", max_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Should be roughly linear
    mid = lrs[49]
    assert 0.4 < mid < 0.6, f"Linear midpoint should be ~0.5, got {mid}"


def test_wsd_scheduler():
    """WSD scheduler: warmup → stable → decay."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, "wsd", max_steps=100,
                                warmup_steps=10, stable_ratio=0.6)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # After warmup, should be at peak during stable phase
    assert lrs[30] > 0.9, f"Stable phase LR should be ~1.0, got {lrs[30]}"
    # At end, should be decayed
    assert lrs[-1] < 0.2, f"Decay phase end LR should be low, got {lrs[-1]}"
