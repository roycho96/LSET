"""Tests for fused GeGLU activation kernel."""

import pytest
import torch
import torch.nn.functional as F

from lset.kernels.geglu import fused_geglu
from lset.kernels.geglu import geglu


def _reference_geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference implementation: gelu_tanh(gate) * up."""
    return F.gelu(gate, approximate="tanh") * up


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestFusedGeGLU:
    """Tests for the Triton fused GeGLU kernel."""

    def test_numerical_match_bf16(self, device):
        gate = torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
        up = torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
        expected = _reference_geglu(gate, up)
        result = fused_geglu(gate, up)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_numerical_match_fp32(self, device):
        gate = torch.randn(128, 1024, device=device, dtype=torch.float32)
        up = torch.randn(128, 1024, device=device, dtype=torch.float32)
        expected = _reference_geglu(gate, up)
        result = fused_geglu(gate, up)
        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_3d_tensor(self, device):
        B, S, D = 4, 32, 1024
        gate = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        up = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        expected = _reference_geglu(gate, up)
        result = fused_geglu(gate, up)
        assert result.shape == (B, S, D)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_gradient_correctness(self, device):
        gate = torch.randn(64, 512, device=device, dtype=torch.float32, requires_grad=True)
        up = torch.randn(64, 512, device=device, dtype=torch.float32, requires_grad=True)
        grad_out = torch.randn(64, 512, device=device, dtype=torch.float32)

        # Reference
        gate_ref = gate.clone().detach().requires_grad_(True)
        up_ref = up.clone().detach().requires_grad_(True)
        y_ref = _reference_geglu(gate_ref, up_ref)
        y_ref.backward(grad_out)

        # Fused
        y_fused = fused_geglu(gate, up)
        y_fused.backward(grad_out)

        assert torch.allclose(gate.grad, gate_ref.grad, atol=1e-5, rtol=1e-5), (
            f"gate grad mismatch: max diff {(gate.grad - gate_ref.grad).abs().max():.2e}"
        )
        assert torch.allclose(up.grad, up_ref.grad, atol=1e-5, rtol=1e-5), (
            f"up grad mismatch: max diff {(up.grad - up_ref.grad).abs().max():.2e}"
        )

    def test_gradcheck_fp64(self, device):
        gate = torch.randn(16, 64, device=device, dtype=torch.float64, requires_grad=True)
        up = torch.randn(16, 64, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            fused_geglu,
            (gate, up),
            eps=1e-5,
            atol=1e-3,
            fast_mode=True,
        )

    def test_auto_dispatch_cpu_fallback(self):
        gate = torch.randn(128, 1024, dtype=torch.float32)
        up = torch.randn(128, 1024, dtype=torch.float32)
        expected = _reference_geglu(gate, up)
        result = geglu(gate, up)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_gradcache_no_fill_overhead(self, device):
        """Verify gate/up gradients flow through standard autograd, not custom Function internals.

        The FusedGeGLU Function takes gate and up as leaf inputs and returns
        the fused output. After backward, gate.grad and up.grad should be
        populated directly (leaf tensors get .grad set by autograd). This
        confirms that weight gradients upstream of gate/up use standard
        AccumulateGrad without the fill_ overhead that would occur if weight
        tensors were passed into the custom Function.
        """
        gate = torch.randn(64, 256, device=device, dtype=torch.float32, requires_grad=True)
        up = torch.randn(64, 256, device=device, dtype=torch.float32, requires_grad=True)

        out = fused_geglu(gate, up)
        loss = out.sum()
        loss.backward()

        # gate and up are leaf tensors, so .grad should be populated
        assert gate.grad is not None, "gate.grad is None after backward"
        assert up.grad is not None, "up.grad is None after backward"
        assert gate.grad.shape == gate.shape
        assert up.grad.shape == up.shape
