"""Tests for checkpoint save/load."""

import os
import tempfile

import torch
import torch.nn as nn

from lset.core.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip():
    """Save and load checkpoint, verify model outputs match."""
    torch.manual_seed(42)
    model = nn.Linear(64, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do a step
    x = torch.randn(4, 64)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    out_before = model(x).detach().clone()

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, optimizer, step=10, output_dir=tmpdir)

        # Corrupt model
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

        out_corrupted = model(x).detach()
        assert not torch.allclose(out_before, out_corrupted)

        # Load
        checkpoint_dir = os.path.join(tmpdir, "step_10")
        step = load_checkpoint(model, optimizer, checkpoint_dir)

        assert step == 10
        out_after = model(x).detach()
        assert torch.allclose(out_before, out_after, atol=1e-6)


def test_optimizer_state_restored():
    """Verify optimizer state is restored correctly."""
    torch.manual_seed(42)
    model = nn.Linear(32, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do a few steps to build up optimizer state
    for _ in range(5):
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Get optimizer state
    state_before = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in optimizer.state[list(optimizer.state.keys())[0]].items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, optimizer, step=5, output_dir=tmpdir)

        # New model + optimizer
        model2 = nn.Linear(32, 32)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        checkpoint_dir = os.path.join(tmpdir, "step_5")
        load_checkpoint(model2, optimizer2, checkpoint_dir)

        state_after = {k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in optimizer2.state[list(optimizer2.state.keys())[0]].items()}

        for key in state_before:
            if isinstance(state_before[key], torch.Tensor):
                assert torch.allclose(state_before[key], state_after[key])
