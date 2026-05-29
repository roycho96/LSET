"""Tests for fused_double_residual_rms_norm — parity + autograd."""

import pytest
import torch

from lset.kernels.double_residual_rmsnorm import fused_double_residual_rms_norm


def _ref(a, r, w1, w2, eps=1e-6):
    def rms_norm(x, w):
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (w * (xf * torch.rsqrt(var + eps))).to(dtype)

    z = rms_norm(a, w1)
    nr = r + z
    out = rms_norm(nr, w2)
    return out, nr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_forward_matches_reference():
    N, D = 512, 1024
    dtype = torch.bfloat16
    a = torch.randn(N, D, device="cuda", dtype=dtype)
    r = torch.randn(N, D, device="cuda", dtype=dtype)
    w1 = torch.randn(D, device="cuda", dtype=dtype)
    w2 = torch.randn(D, device="cuda", dtype=dtype)

    out_ref, nr_ref = _ref(a, r, w1, w2)
    out, nr = fused_double_residual_rms_norm(a, r, w1, w2)

    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(nr, nr_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_forward_3d_shape():
    B, S, D = 2, 128, 512
    dtype = torch.bfloat16
    a = torch.randn(B, S, D, device="cuda", dtype=dtype)
    r = torch.randn(B, S, D, device="cuda", dtype=dtype)
    w1 = torch.randn(D, device="cuda", dtype=dtype)
    w2 = torch.randn(D, device="cuda", dtype=dtype)

    out_ref, nr_ref = _ref(a, r, w1, w2)
    out, nr = fused_double_residual_rms_norm(a, r, w1, w2)

    assert out.shape == (B, S, D)
    assert nr.shape == (B, S, D)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(nr, nr_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_backward_matches_reference():
    N, D = 256, 512
    dtype = torch.bfloat16
    a = torch.randn(N, D, device="cuda", dtype=dtype, requires_grad=True)
    r = torch.randn(N, D, device="cuda", dtype=dtype, requires_grad=True)
    w1 = torch.randn(D, device="cuda", dtype=dtype, requires_grad=True)
    w2 = torch.randn(D, device="cuda", dtype=dtype, requires_grad=True)

    out_ref, nr_ref = _ref(a, r, w1, w2)
    (out_ref.sum() + nr_ref.sum()).backward()
    da_ref = a.grad.clone()
    dr_ref = r.grad.clone()
    dw1_ref = w1.grad.clone()
    dw2_ref = w2.grad.clone()

    a.grad = None
    r.grad = None
    w1.grad = None
    w2.grad = None

    out, nr = fused_double_residual_rms_norm(a, r, w1, w2)
    (out.sum() + nr.sum()).backward()

    # dW accumulates over rows in bf16, so its tolerance is looser.
    torch.testing.assert_close(a.grad, da_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(r.grad, dr_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(w1.grad, dw1_ref, atol=3e-1, rtol=3e-1)
    torch.testing.assert_close(w2.grad, dw2_ref, atol=3e-1, rtol=3e-1)
