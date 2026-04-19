"""Fused Q+K RoPE parity + autograd check."""

import pytest
import torch

from lset.kernels.rope import FusedRoPEQK
from lset.kernels.rope import apply_rotary_pos_emb


def _reference_rope(q, k, cos, sin):
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _make_cos_sin(seq, head_dim, device, dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_rope_padded_matches_reference():
    B, H_q, H_k, S, D = 2, 8, 2, 256, 128
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H_k, S, D, device=device, dtype=dtype)
    cos_2d, sin_2d = _make_cos_sin(S, D, device, dtype)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    q_ref, k_ref = _reference_rope(q, k, cos, sin)
    q_out, k_out = FusedRoPEQK.apply(q, k, cos, sin)

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_rope_packed_matches_reference():
    T, H_q, H_k, D = 512, 8, 2, 128
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(T, H_q, D, device=device, dtype=dtype)
    k = torch.randn(T, H_k, D, device=device, dtype=dtype)
    cos_2d, sin_2d = _make_cos_sin(T, D, device, dtype)
    cos = cos_2d.unsqueeze(1)  # (T, 1, D)
    sin = sin_2d.unsqueeze(1)

    q_ref, k_ref = _reference_rope(q, k, cos, sin)
    q_out, k_out = FusedRoPEQK.apply(q, k, cos, sin)

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_fused_rope_autograd_matches_reference():
    B, H_q, H_k, S, D = 1, 4, 2, 128, 64
    dtype = torch.float32
    device = "cuda"

    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H_k, S, D, device=device, dtype=dtype, requires_grad=True)
    cos_2d, sin_2d = _make_cos_sin(S, D, device, dtype)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    q_ref, k_ref = _reference_rope(q, k, cos, sin)
    (q_ref.sum() + k_ref.sum()).backward()
    q_grad_ref = q.grad.clone()
    k_grad_ref = k.grad.clone()

    q.grad = None
    k.grad = None
    q_out, k_out = FusedRoPEQK.apply(q, k, cos, sin)
    (q_out.sum() + k_out.sum()).backward()

    torch.testing.assert_close(q.grad, q_grad_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad, k_grad_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_dispatch_small_falls_back_to_eager():
    # Below threshold → eager path (CPU-compatible but we're on CUDA for parity)
    B, H_q, H_k, S, D = 1, 4, 2, 32, 64
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.float32)
    cos_2d, sin_2d = _make_cos_sin(S, D, "cuda", torch.float32)
    cos = cos_2d.unsqueeze(0).unsqueeze(0)
    sin = sin_2d.unsqueeze(0).unsqueeze(0)

    q_ref, k_ref = _reference_rope(q, k, cos, sin)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)
