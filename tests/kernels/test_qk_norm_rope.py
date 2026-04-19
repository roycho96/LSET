"""Fused QK-Norm + RoPE parity + autograd."""

import pytest
import torch

from lset.kernels.qk_norm_rope import fused_qk_norm_rope
from lset.kernels.qk_norm_rope import fused_rope_only
from lset.kernels.rmsnorm import fused_rms_norm
from lset.kernels.rope import fused_apply_rotary_pos_emb


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _ref_rms_norm(x, w, eps=1e-6):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (w * (x_f * torch.rsqrt(var + eps))).to(x.dtype)


def _make_cos_sin(seq, head_dim, device, dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_qk_norm_rope_padded_matches_reference():
    B, H_q, H_k, S, D = 2, 8, 2, 256, 128
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H_k, S, D, device=device, dtype=dtype)
    q_w = torch.randn(D, device=device, dtype=dtype)
    k_w = torch.randn(D, device=device, dtype=dtype)
    cos_2d, sin_2d = _make_cos_sin(S, D, device, dtype)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    q_ref = _ref_rms_norm(q, q_w, eps=1e-6)
    k_ref = _ref_rms_norm(k, k_w, eps=1e-6)
    q_ref, k_ref = _ref_rope(q_ref, k_ref, cos, sin)

    q_out, k_out = fused_qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6)

    torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_qk_norm_rope_packed_matches_reference():
    T, H_q, H_k, D = 512, 8, 2, 128
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(T, H_q, D, device=device, dtype=dtype)
    k = torch.randn(T, H_k, D, device=device, dtype=dtype)
    q_w = torch.randn(D, device=device, dtype=dtype)
    k_w = torch.randn(D, device=device, dtype=dtype)
    cos_2d, sin_2d = _make_cos_sin(T, D, device, dtype)
    cos = cos_2d.unsqueeze(1)  # (T, 1, D)
    sin = sin_2d.unsqueeze(1)

    q_ref = _ref_rms_norm(q, q_w, eps=1e-6)
    k_ref = _ref_rms_norm(k, k_w, eps=1e-6)
    q_ref, k_ref = _ref_rope(q_ref, k_ref, cos, sin)

    q_out, k_out = fused_qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6)

    torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_rope_only_matches_rope_kernel():
    B, H_q, H_k, S, D = 1, 4, 2, 128, 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H_k, S, D, device=device, dtype=dtype)
    cos_2d, sin_2d = _make_cos_sin(S, D, device, dtype)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    q_ref, k_ref = fused_apply_rotary_pos_emb(q, k, cos, sin)
    q_out, k_out = fused_rope_only(q, k, cos, sin)

    # bf16 roundings differ slightly between the two kernels (extra fp32 cast
    # in the shared kernel); widen tolerance accordingly.
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_qk_norm_rope_autograd():
    B, H_q, H_k, S, D = 1, 4, 2, 128, 64
    dtype = torch.float32
    device = "cuda"

    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H_k, S, D, device=device, dtype=dtype, requires_grad=True)
    q_w = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    k_w = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    cos_2d, sin_2d = _make_cos_sin(S, D, device, dtype)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    # Reference path via separate RMSNorm + RoPE.
    q_ref = _ref_rms_norm(q, q_w, eps=1e-6)
    k_ref = _ref_rms_norm(k, k_w, eps=1e-6)
    q_ref_out, k_ref_out = _ref_rope(q_ref, k_ref, cos, sin)
    (q_ref_out.sum() + k_ref_out.sum()).backward()
    q_grad_ref = q.grad.clone()
    k_grad_ref = k.grad.clone()
    qw_grad_ref = q_w.grad.clone()
    kw_grad_ref = k_w.grad.clone()

    q.grad = None
    k.grad = None
    q_w.grad = None
    k_w.grad = None

    q_out, k_out = fused_qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6)
    (q_out.sum() + k_out.sum()).backward()

    torch.testing.assert_close(q.grad, q_grad_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad, k_grad_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(q_w.grad, qw_grad_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_w.grad, kw_grad_ref, atol=1e-4, rtol=1e-4)
