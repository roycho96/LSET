"""Tests for FusedLayerNorm Triton kernel."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _reference_layer_norm(x, weight, bias, eps=1e-5):
    """Reference: torch.nn.functional.layer_norm."""
    return F.layer_norm(x, weight.shape, weight, bias, eps)


class TestFusedLayerNorm:

    def test_numerical_match_bf16(self, device):
        """Fused vs F.layer_norm reference in bf16."""
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 128, 768
        x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)
        bias = torch.randn(D, device=device, dtype=torch.bfloat16)

        out_fused = fused_layer_norm(x, weight, bias)
        out_ref = _reference_layer_norm(x, weight, bias)

        torch.testing.assert_close(out_fused, out_ref, atol=5e-2, rtol=5e-2)

    def test_numerical_match_fp32(self, device):
        """Fused vs F.layer_norm reference in fp32."""
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 64, 512
        x = torch.randn(N, D, device=device, dtype=torch.float32)
        weight = torch.randn(D, device=device, dtype=torch.float32)
        bias = torch.randn(D, device=device, dtype=torch.float32)

        out_fused = fused_layer_norm(x, weight, bias)
        out_ref = _reference_layer_norm(x, weight, bias)

        torch.testing.assert_close(out_fused, out_ref, atol=1e-5, rtol=1e-5)

    def test_3d_tensor(self, device):
        """Works with (B, S, D) inputs."""
        from lset.kernels.layernorm import fused_layer_norm

        B, S, D = 2, 64, 768
        x = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)
        bias = torch.randn(D, device=device, dtype=torch.bfloat16)

        out_fused = fused_layer_norm(x, weight, bias)
        out_ref = _reference_layer_norm(x, weight, bias)

        assert out_fused.shape == (B, S, D)
        torch.testing.assert_close(out_fused, out_ref, atol=5e-2, rtol=5e-2)

    def test_gradient_correctness_x(self, device):
        """Gradient for x matches reference in fp32."""
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 64, 256
        x_fused = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        x_ref = x_fused.detach().clone().requires_grad_(True)
        weight = torch.randn(D, device=device, dtype=torch.float32)
        bias = torch.randn(D, device=device, dtype=torch.float32)
        grad_out = torch.randn(N, D, device=device, dtype=torch.float32)

        out_fused = fused_layer_norm(x_fused, weight, bias)
        out_fused.backward(grad_out)

        out_ref = _reference_layer_norm(x_ref, weight, bias)
        out_ref.backward(grad_out)

        torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

    def test_gradient_weight_bias(self, device):
        """Gradients for weight and bias match reference in fp32.

        Weight and bias are applied via standard PyTorch ops outside the custom
        autograd Function, so their gradients flow through normal autograd.
        """
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 64, 256
        x = torch.randn(N, D, device=device, dtype=torch.float32)
        weight_fused = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
        bias_fused = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
        weight_ref = weight_fused.detach().clone().requires_grad_(True)
        bias_ref = bias_fused.detach().clone().requires_grad_(True)
        grad_out = torch.randn(N, D, device=device, dtype=torch.float32)

        out_fused = fused_layer_norm(x, weight_fused, bias_fused)
        out_fused.backward(grad_out)

        out_ref = _reference_layer_norm(x, weight_ref, bias_ref)
        out_ref.backward(grad_out)

        torch.testing.assert_close(weight_fused.grad, weight_ref.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(bias_fused.grad, bias_ref.grad, atol=1e-5, rtol=1e-5)

    def test_gradcheck_fp64(self, device):
        """torch.autograd.gradcheck with fp64."""
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 8, 32
        x = torch.randn(N, D, device=device, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(D, device=device, dtype=torch.float64, requires_grad=True)

        def fn(x_, w_, b_):
            return fused_layer_norm(x_, w_, b_, eps=1e-5)

        assert torch.autograd.gradcheck(fn, (x, weight, bias), eps=1e-6, atol=1e-4)

    def test_auto_dispatch_cpu_fallback(self):
        """CPU inputs use F.layer_norm fallback via layer_norm()."""
        from lset.kernels.layernorm import layer_norm

        N, D = 64, 256
        x = torch.randn(N, D, dtype=torch.float32)
        weight = torch.randn(D, dtype=torch.float32)
        bias = torch.randn(D, dtype=torch.float32)

        out = layer_norm(x, weight, bias)
        out_ref = _reference_layer_norm(x, weight, bias)

        torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-6)

    def test_gradcache_weight_bias_outside_function(self, device):
        """Verify weight.grad and bias.grad exist after backward.

        Since weight and bias are applied OUTSIDE the custom autograd Function,
        their gradients flow through standard autograd (no AccumulateGrad fill_
        overhead in GradCache chunked backward).
        """
        from lset.kernels.layernorm import fused_layer_norm

        N, D = 128, 256
        x = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

        out = fused_layer_norm(x, weight, bias)
        loss = out.sum()
        loss.backward()

        # Weight and bias must have gradients
        assert weight.grad is not None, "weight.grad should exist (outside custom Function)"
        assert bias.grad is not None, "bias.grad should exist (outside custom Function)"
        assert x.grad is not None, "x.grad should exist"

        # Gradients should be non-zero
        assert weight.grad.abs().sum() > 0, "weight.grad should be non-zero"
        assert bias.grad.abs().sum() > 0, "bias.grad should be non-zero"

        # Weight grad shape should match weight
        assert weight.grad.shape == weight.shape
        assert bias.grad.shape == bias.shape
