"""Tests for FusedResidualLayerNorm Triton kernel.

Covers: numerical match (bf16/fp32), 3-D tensors, gradient correctness
(inputs + weight/bias), gradcheck (fp64), CPU fallback, residual identity.
"""

import pytest
import torch
import torch.nn.functional as F

from lset.kernels.fused_residual_layernorm import (
    fused_residual_layer_norm,
    residual_layer_norm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference(residual, attn_out, weight, bias, eps=1e-5):
    new_residual = residual + attn_out
    normed = F.layer_norm(new_residual, weight.shape, weight, bias, eps)
    return normed, new_residual


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# 1. Numerical match — bf16
# ---------------------------------------------------------------------------

def test_numerical_match_bf16(device):
    N, D = 128, 768
    torch.manual_seed(42)
    residual = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    attn_out = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weight = torch.randn(D, device=device, dtype=torch.bfloat16)
    bias = torch.randn(D, device=device, dtype=torch.bfloat16)

    normed, new_res = fused_residual_layer_norm(residual, attn_out, weight, bias)
    ref_normed, ref_res = _reference(residual, attn_out, weight, bias)

    torch.testing.assert_close(normed, ref_normed, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(new_res, ref_res, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 2. Numerical match — fp32
# ---------------------------------------------------------------------------

def test_numerical_match_fp32(device):
    N, D = 128, 768
    torch.manual_seed(0)
    residual = torch.randn(N, D, device=device, dtype=torch.float32)
    attn_out = torch.randn(N, D, device=device, dtype=torch.float32)
    weight = torch.randn(D, device=device, dtype=torch.float32)
    bias = torch.randn(D, device=device, dtype=torch.float32)

    normed, new_res = fused_residual_layer_norm(residual, attn_out, weight, bias)
    ref_normed, ref_res = _reference(residual, attn_out, weight, bias)

    torch.testing.assert_close(normed, ref_normed, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new_res, ref_res, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 3. 3-D tensor (B, S, D)
# ---------------------------------------------------------------------------

def test_3d_tensor(device):
    B, S, D = 2, 64, 768
    torch.manual_seed(7)
    residual = torch.randn(B, S, D, device=device, dtype=torch.float32)
    attn_out = torch.randn(B, S, D, device=device, dtype=torch.float32)
    weight = torch.randn(D, device=device, dtype=torch.float32)
    bias = torch.randn(D, device=device, dtype=torch.float32)

    normed, new_res = fused_residual_layer_norm(residual, attn_out, weight, bias)
    ref_normed, ref_res = _reference(residual, attn_out, weight, bias)

    assert normed.shape == (B, S, D)
    assert new_res.shape == (B, S, D)
    torch.testing.assert_close(normed, ref_normed, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new_res, ref_res, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 4. Gradient correctness — residual and attn_out
# ---------------------------------------------------------------------------

def test_gradient_correctness(device):
    N, D = 64, 256
    torch.manual_seed(1)

    def _run(fn):
        r = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        a = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=True)
        w = torch.randn(D, device=device, dtype=torch.float32, requires_grad=False)
        b = torch.randn(D, device=device, dtype=torch.float32, requires_grad=False)
        normed, _ = fn(r, a, w, b)
        loss = normed.sum()
        loss.backward()
        return r.grad.clone(), a.grad.clone()

    # Seed must be identical for both runs
    torch.manual_seed(1)
    grad_r_fused, grad_a_fused = _run(fused_residual_layer_norm)
    torch.manual_seed(1)
    grad_r_ref, grad_a_ref = _run(_reference)

    torch.testing.assert_close(grad_r_fused, grad_r_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_a_fused, grad_a_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Gradient correctness — weight and bias
# ---------------------------------------------------------------------------

def test_gradient_weight_bias(device):
    N, D = 64, 256
    torch.manual_seed(2)

    def _run(fn):
        r = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=False)
        a = torch.randn(N, D, device=device, dtype=torch.float32, requires_grad=False)
        w = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
        normed, _ = fn(r, a, w, b)
        loss = normed.sum()
        loss.backward()
        return w.grad.clone(), b.grad.clone()

    torch.manual_seed(2)
    grad_w_fused, grad_b_fused = _run(fused_residual_layer_norm)
    torch.manual_seed(2)
    grad_w_ref, grad_b_ref = _run(_reference)

    torch.testing.assert_close(grad_w_fused, grad_w_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_b_fused, grad_b_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. gradcheck — fp64
# ---------------------------------------------------------------------------

def test_gradcheck_fp64(device):
    N, D = 8, 32
    torch.manual_seed(3)
    residual = torch.randn(N, D, device=device, dtype=torch.float64, requires_grad=True)
    attn_out = torch.randn(N, D, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(D, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=torch.float64, requires_grad=True)
    eps = 1e-5

    def _fn(r, a):
        normed, _ = fused_residual_layer_norm(r, a, weight, bias, eps)
        return normed

    assert torch.autograd.gradcheck(_fn, (residual, attn_out), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Auto-dispatch CPU fallback
# ---------------------------------------------------------------------------

def test_auto_dispatch_cpu_fallback():
    N, D = 512, 128
    torch.manual_seed(4)
    residual = torch.randn(N, D, dtype=torch.float32)
    attn_out = torch.randn(N, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)

    normed, new_res = residual_layer_norm(residual, attn_out, weight, bias)
    ref_normed, ref_res = _reference(residual, attn_out, weight, bias)

    torch.testing.assert_close(normed, ref_normed, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(new_res, ref_res, atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# 8. new_residual is exactly residual + attn_out
# ---------------------------------------------------------------------------

def test_new_residual_is_sum(device):
    N, D = 256, 512
    torch.manual_seed(5)
    residual = torch.randn(N, D, device=device, dtype=torch.float32)
    attn_out = torch.randn(N, D, device=device, dtype=torch.float32)
    weight = torch.ones(D, device=device, dtype=torch.float32)
    bias = torch.zeros(D, device=device, dtype=torch.float32)

    _, new_res = fused_residual_layer_norm(residual, attn_out, weight, bias)
    expected = residual + attn_out

    torch.testing.assert_close(new_res, expected, atol=0, rtol=0)
