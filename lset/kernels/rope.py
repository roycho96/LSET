"""Fused Q+K Rotary Position Embedding — single Triton kernel.

One program per token handles Q and K together. ``cos/sin`` are loaded once
from HBM and shared across all Q heads (``n_qh``) and K heads (``n_kh``) via
a 2D register tile ``(pad_n_qh, pad_hd/2)``.

The kernel writes in-place into a freshly-allocated contiguous buffer (the
output of ``transpose(1, 2).contiguous()`` or ``clone()``). Callers never see
their input mutated — we always own the buffer we write to. This matches
the Liger-Kernel pattern and avoids the extra ``empty_like`` + post-kernel
``contiguous()`` round-trip used by the previous LSET wrapper.

Shape support (cos/sin is broadcast along the leading dims via ``pid % S_cos``):
  - 4D padded:  q/k ``(B, H, S, D)``, cos/sin ``(1, 1, S, D)``
  - 3D packed:  q/k ``(T, H, D)``,   cos/sin ``(T, 1, D)``
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from torch import Tensor


@triton.jit
def _rope_qk_kernel(
    Q, q_token_stride,
    K, k_token_stride,
    COS, SIN, cos_token_stride,
    N,
    S_COS,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    """Combined Q/K RoPE, in-place on Q and K.

      Forward:  y1 = x1*c − x2*s, y2 = x1*s + x2*c
      Backward: dx1 = dy1*c + dy2*s, dx2 = −dy1*s + dy2*c
    """
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    tok_cos = pid % S_COS
    c_base = COS + tok_cos * cos_token_stride
    s_base = SIN + tok_cos * cos_token_stride

    dh = tl.arange(0, pad_hd_half)
    dh_mask = dh < (hd // 2)
    cos_row = tl.load(c_base + dh, mask=dh_mask, other=0.0)
    sin_row = tl.load(s_base + dh, mask=dh_mask, other=0.0)

    # --- Q ---
    q_base = Q + pid * q_token_stride
    q_row = tl.arange(0, pad_n_qh)[:, None]
    q_col = dh[None, :]
    q_off = q_row * hd + q_col
    q_mask = (q_row < n_qh) & (q_col < (hd // 2))

    q1 = tl.load(q_base + q_off, mask=q_mask, other=0.0).to(sin_row.dtype)
    q2 = tl.load(q_base + q_off + (hd // 2), mask=q_mask, other=0.0).to(sin_row.dtype)
    if BACKWARD_PASS:
        q1_new = q1 * cos_row + q2 * sin_row
        q2_new = -q1 * sin_row + q2 * cos_row
    else:
        q1_new = q1 * cos_row - q2 * sin_row
        q2_new = q1 * sin_row + q2 * cos_row
    tl.store(q_base + q_off, q1_new, mask=q_mask)
    tl.store(q_base + q_off + (hd // 2), q2_new, mask=q_mask)

    # --- K ---
    k_base = K + pid * k_token_stride
    k_row = tl.arange(0, pad_n_kh)[:, None]
    k_col = dh[None, :]
    k_off = k_row * hd + k_col
    k_mask = (k_row < n_kh) & (k_col < (hd // 2))

    k1 = tl.load(k_base + k_off, mask=k_mask, other=0.0).to(sin_row.dtype)
    k2 = tl.load(k_base + k_off + (hd // 2), mask=k_mask, other=0.0).to(sin_row.dtype)
    if BACKWARD_PASS:
        k1_new = k1 * cos_row + k2 * sin_row
        k2_new = -k1 * sin_row + k2 * cos_row
    else:
        k1_new = k1 * cos_row - k2 * sin_row
        k2_new = k1 * sin_row + k2 * cos_row
    tl.store(k_base + k_off, k1_new, mask=k_mask)
    tl.store(k_base + k_off + (hd // 2), k2_new, mask=k_mask)


def _launch_rope_qk(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, backward: bool):
    """Run ``_rope_qk_kernel`` over Q and K in-place on a fresh buffer.

    For 4D ``(B, H, S, D)`` we first ``transpose(1, 2).contiguous()`` to the
    physical ``(B, S, H, D)`` layout — ``transpose`` makes the tensor
    non-contiguous so ``contiguous()`` always allocates a fresh buffer we
    own. For 3D ``(N, H, D)`` inputs that are already contiguous, we clone
    so we never mutate the caller's tensor.
    """
    is_padded = q.dim() == 4
    if is_padded:
        B, H_q, S, D = q.shape
        H_k = k.shape[1]
        q_work = q.transpose(1, 2).contiguous()  # fresh (B, S, H_q, D)
        k_work = k.transpose(1, 2).contiguous()
        N = B * S
    else:
        N, H_q, D = q.shape
        H_k = k.shape[1]
        q_work = q.contiguous()
        if q_work is q:
            q_work = q_work.clone()
        k_work = k.contiguous()
        if k_work is k:
            k_work = k_work.clone()

    cos2d = cos.reshape(-1, cos.shape[-1]).contiguous()
    sin2d = sin.reshape(-1, sin.shape[-1]).contiguous()
    S_cos = cos2d.shape[0]

    # Token stride is H*D for both layouts: stride(1) on (B, S, H, D) contig,
    # stride(0) on (N, H, D) contig.
    q_token_stride = q_work.stride(1) if is_padded else q_work.stride(0)
    k_token_stride = k_work.stride(1) if is_padded else k_work.stride(0)

    pad_hd_half = max(16, triton.next_power_of_2(D // 2))
    pad_n_qh = max(1, triton.next_power_of_2(H_q))
    pad_n_kh = max(1, triton.next_power_of_2(H_k))

    _rope_qk_kernel[(N,)](
        q_work, q_token_stride,
        k_work, k_token_stride,
        cos2d, sin2d, cos2d.stride(0),
        N, S_cos,
        H_q, H_k, D,
        pad_n_qh, pad_n_kh, pad_hd_half,
        BACKWARD_PASS=backward,
        num_warps=4,
    )

    if is_padded:
        # Return the transposed view — downstream SDPA/FA handles non-contig
        # input without extra copies. If a caller truly needs contig they can
        # call .contiguous() themselves, matching Liger-Kernel's API.
        return q_work.transpose(1, 2), k_work.transpose(1, 2)
    return q_work, k_work


class FusedRoPEQK(torch.autograd.Function):
    """Combined Q/K RoPE autograd Function.

    Forward and backward both use ``_rope_qk_kernel`` with a ``BACKWARD_PASS``
    flag — no saved activations beyond cos/sin.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out, k_out = _launch_rope_qk(q, k, cos, sin, backward=False)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        dq, dk = _launch_rope_qk(grad_q, grad_k, cos, sin, backward=True)
        return dq, dk, None, None


_FUSED_ROPE_THRESHOLD = 128


def fused_apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Apply fused RoPE to Q and K in a single kernel launch."""
    return FusedRoPEQK.apply(q, k, cos, sin)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """RoPE with automatic Triton dispatch.

    Falls back to the eager ``rotate_half`` formula on CPU or very small
    inputs where the kernel launch overhead dominates.
    """
    if q.is_cuda:
        T = q.shape[0] if q.dim() == 3 else q.shape[0] * q.shape[2]
        if T >= _FUSED_ROPE_THRESHOLD:
            return fused_apply_rotary_pos_emb(q, k, cos, sin)

    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
