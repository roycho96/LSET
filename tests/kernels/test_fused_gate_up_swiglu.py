"""Fused gate_up + SwiGLU — parity + autograd tests."""

import pytest
import torch
import torch.nn.functional as F

from lset.kernels.fused_gate_up_swiglu import FusedGateUpSwiGLU
from lset.kernels.fused_gate_up_swiglu import fused_gate_up_swiglu


def _ref(x, w):
    gu = F.linear(x, w)
    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_forward_matches_reference_2d():
    M, K, I = 128, 512, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2 * I, K, device="cuda", dtype=torch.bfloat16)

    ref = _ref(x, w)
    out = fused_gate_up_swiglu(x, w)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_forward_matches_reference_3d():
    B, S, K, I = 2, 64, 512, 256
    x = torch.randn(B, S, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2 * I, K, device="cuda", dtype=torch.bfloat16)

    ref = _ref(x, w)
    out = fused_gate_up_swiglu(x, w)

    assert out.shape == (B, S, I)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_backward_matches_reference():
    """Parity at bf16 (the production dtype)."""
    M, K, I = 64, 256, 128
    dtype = torch.bfloat16
    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(2 * I, K, device="cuda", dtype=dtype, requires_grad=True)

    # Compare the fused path against the unfused F.linear + swiglu path.
    gu = F.linear(x, w)
    gate, up = gu.chunk(2, -1)
    ref = F.silu(gate) * up
    ref.sum().backward()
    dx_ref = x.grad.clone()
    dw_ref = w.grad.clone()

    x.grad = None
    w.grad = None

    out = fused_gate_up_swiglu(x, w)
    out.sum().backward()

    torch.testing.assert_close(x.grad, dx_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(w.grad, dw_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_module_wrapper_parity():
    M, K, I = 128, 512, 256
    mod = FusedGateUpSwiGLU(K, I).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Fusion on
    out_fused = mod(x)

    # Fallback
    mod.disable_fusion()
    out_eager = mod(x)

    torch.testing.assert_close(out_fused, out_eager, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_module_preserves_linear_attrs():
    """TP/FP8 filters key on .weight, .in_features, .out_features."""
    mod = FusedGateUpSwiGLU(512, 256)
    assert mod.in_features == 512
    assert mod.out_features == 512  # = 2*I so TP sees the right shard dim
    assert mod.weight.shape == (512, 512)
