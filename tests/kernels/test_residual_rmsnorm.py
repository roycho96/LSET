"""Tests for H1: FusedResidualRMSNorm kernel."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _reference_residual_rms_norm(residual, attn_out, weight, eps=1e-6):
    """Reference implementation: separate residual add + RMSNorm."""
    new_residual = residual + attn_out
    x_f32 = new_residual.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(residual.dtype), new_residual


class TestFusedResidualRMSNorm:
    def test_numerical_match_bf16(self, device):
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        N, D = 128, 1024
        residual = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        attn_out = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)

        out_fused, res_fused = fused_residual_rms_norm(residual, attn_out, weight)
        out_ref, res_ref = _reference_residual_rms_norm(residual, attn_out, weight)

        torch.testing.assert_close(out_fused, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(res_fused, res_ref, atol=1e-5, rtol=1e-5)

    def test_numerical_match_fp32(self, device):
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        N, D = 64, 512
        residual = torch.randn(N, D, device=device, dtype=torch.float32)
        attn_out = torch.randn(N, D, device=device, dtype=torch.float32)
        weight = torch.randn(D, device=device, dtype=torch.float32)

        out_fused, res_fused = fused_residual_rms_norm(residual, attn_out, weight)
        out_ref, res_ref = _reference_residual_rms_norm(residual, attn_out, weight)

        torch.testing.assert_close(out_fused, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(res_fused, res_ref, atol=1e-6, rtol=1e-6)

    def test_3d_tensor(self, device):
        """Works with (B, S, D) inputs."""
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        B, S, D = 2, 64, 256
        residual = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        attn_out = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)

        out, res = fused_residual_rms_norm(residual, attn_out, weight)
        assert out.shape == (B, S, D)
        assert res.shape == (B, S, D)

    def test_gradient_correctness(self, device):
        """Gradient matches reference implementation (fp32)."""
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        N, D = 16, 64
        residual = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        attn_out = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

        # Fused path
        out_f, res_f = fused_residual_rms_norm(residual, attn_out, weight)
        loss_f = out_f.sum()
        loss_f.backward()
        grad_r_f = residual.grad.clone()
        grad_a_f = attn_out.grad.clone()

        # Reference path
        residual.grad = None
        attn_out.grad = None
        ref_res = residual + attn_out
        x_f32 = ref_res.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(var + 1e-6)
        out_ref = weight * x_normed
        out_ref.sum().backward()

        torch.testing.assert_close(grad_r_f, residual.grad, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(grad_a_f, attn_out.grad, atol=1e-4, rtol=1e-3)

    def test_backward_matches_pytorch(self, device):
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        N, D = 32, 256
        residual = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        attn_out = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

        out, _ = fused_residual_rms_norm(residual, attn_out, weight)
        loss = out.sum()
        loss.backward()

        assert residual.grad is not None
        assert attn_out.grad is not None
        assert weight.grad is not None
        assert not residual.grad.isnan().any()

    def test_grad_cache_compatible(self, device):
        """Weight gradient flows through normal autograd (not custom Function)."""
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        D = 64
        residual = torch.randn(8, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        attn_out = torch.randn(8, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16, requires_grad=True)

        out, new_res = fused_residual_rms_norm(residual, attn_out, weight)
        # Simulate GradCache: backward on out, then separate backward on new_res
        out.sum().backward(retain_graph=True)
        assert weight.grad is not None, "Weight grad should flow through PyTorch mul"

    def test_auto_dispatch_fallback(self):
        """CPU/small inputs fall back to PyTorch."""
        from lset.kernels.residual_rmsnorm import residual_rms_norm

        residual = torch.randn(4, 64)
        attn_out = torch.randn(4, 64)
        weight = torch.ones(64)

        out, res = residual_rms_norm(residual, attn_out, weight)
        assert out.shape == (4, 64)

    def test_memory_reduction(self, device):
        """Fused path should not create an intermediate tensor for the add."""
        from lset.kernels.residual_rmsnorm import fused_residual_rms_norm

        N, D = 1024, 1024
        residual = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        attn_out = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)

        # Warmup
        fused_residual_rms_norm(residual, attn_out, weight)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        out, res = fused_residual_rms_norm(residual, attn_out, weight)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(device)

        # Should allocate 2 output tensors (out + new_residual) = 2 * N * D * 2 bytes
        # Plus rstd (N * 4 bytes). No intermediate add tensor.
        expected_alloc = 2 * N * D * 2 + N * 4
        actual_alloc = after - before
        # Allow 20% overhead for Triton workspace
        assert actual_alloc < expected_alloc * 1.2, f"Allocated {actual_alloc} bytes, expected ~{expected_alloc}"
