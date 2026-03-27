"""Tests for MXFP8 (Microscaling FP8) training — experimental.

MXFP8 is a prototype feature in torchao. These tests validate the API
works on our hardware and document known limitations.
"""

import pytest
import torch
import torch.nn as nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMXFP8:
    def test_apply_mxfp8_training(self):
        """MXLinear modules should be created."""
        from lset.distributed.mx import apply_mxfp8_training
        from torchao.prototype.mx_formats.mx_linear import MXLinear

        model = nn.Sequential(
            nn.Linear(64, 128, bias=False, dtype=torch.bfloat16, device="cuda"),
            nn.Linear(128, 64, bias=False, dtype=torch.bfloat16, device="cuda"),
        )
        apply_mxfp8_training(model, recipe="mxfp8_emulated")
        assert isinstance(model[0], MXLinear)
        assert isinstance(model[1], MXLinear)

    def test_mxfp8_forward(self):
        """MXFP8 forward pass should produce valid output."""
        from lset.distributed.mx import apply_mxfp8_training

        model = nn.Sequential(
            nn.Linear(64, 128, bias=False, dtype=torch.bfloat16, device="cuda"),
            nn.Linear(128, 64, bias=False, dtype=torch.bfloat16, device="cuda"),
        )
        apply_mxfp8_training(model, recipe="mxfp8_emulated")

        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        out = model(x)
        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()

    def test_mxfp8_backward(self):
        """MXFP8 backward pass. All dims must be divisible by block_size=32."""
        from lset.distributed.mx import apply_mxfp8_training

        # Single layer: avoids non-contiguous grad_output issue in stacked models
        model = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="cuda")
        apply_mxfp8_training(model, recipe="mxfp8_emulated")

        x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
        target = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        assert model.weight.grad is not None

    def test_mxfp8_training_steps(self):
        """MXFP8 training steps with single linear (prototype API)."""
        from lset.distributed.mx import apply_mxfp8_training

        model = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="cuda")
        apply_mxfp8_training(model, recipe="mxfp8_emulated")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
            target = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
            out = model(x)
            loss = nn.functional.mse_loss(out.float(), target.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_mxfp8_invalid_recipe(self):
        from lset.distributed.mx import apply_mxfp8_training
        model = nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="Invalid MXFP8 recipe"):
            apply_mxfp8_training(model, recipe="invalid")
