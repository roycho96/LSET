"""RandContext: RNG restore makes dropout deterministic across replays."""

import torch

from lset.train.grad_cache import RandContext


def test_restores_cpu_rng():
    x = torch.empty(4)
    rs = RandContext(x)
    # First "forward": draw from the snapshot point
    a = torch.randn(4)
    # Replay: enter → RNG rewinds to pre-draw state → same numbers
    with rs:
        b = torch.randn(4)
    assert torch.equal(a, b)


def test_does_not_leak_rng_outside():
    """After exit, the outer RNG stream is untouched by the rewind."""
    # Baseline: what RNG produces without any RandContext
    torch.manual_seed(0)
    expected = torch.randn(4)

    # Now with a RandContext that rewinds somewhere in between
    torch.manual_seed(0)
    rs = RandContext()
    with rs:
        _ = torch.randn(4)  # advances inner-fork RNG, not outer
    got = torch.randn(4)
    assert torch.equal(expected, got)


def test_dropout_deterministic_across_replays():
    linear = torch.nn.Linear(8, 8)
    drop = torch.nn.Dropout(p=0.5)
    x = torch.randn(4, 8)

    rs = RandContext(x)
    with rs:
        y1 = drop(linear(x))

    with rs:  # replay
        y2 = drop(linear(x))

    assert torch.equal(y1, y2)
