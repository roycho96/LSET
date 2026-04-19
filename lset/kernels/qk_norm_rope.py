"""Fused QK-Norm + RoPE — single Triton kernel for Qwen3/Gemma-style attention.

Replaces the back-to-back sequence

    q = q_norm(q)          # RMSNorm kernel  (read Q, write Q')
    k = k_norm(k)          # RMSNorm kernel  (read K, write K')
    q, k = rope(q, k, ...) # RoPE kernel     (read Q', K', write Q'', K'')

with a single kernel that reads Q and K once, writes Q'' and K'' once, and
uses shared ``cos/sin`` registers across all Q and K heads.

Layout follows the RoPE kernel in ``rope.py`` (per-token grid, 2D
``(n_heads, hd/2)`` head tile). The norm adds an extra reduction per head
row (``sum(q_l² + q_r²)/hd``) and a weight multiply before the rotation.

Launches saved per attention: **2** (was 3: q_norm + k_norm + rope → 1).
HBM round trips saved: **full Q and K intermediates** never materialized.

The kernel supports a ``DO_NORM`` constexpr flag so Llama-style attention
(no qk_norm) can share the exact same kernel for the rope-only path. The
flag collapses the norm prologue at compile time.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from torch import Tensor

try:
    from triton.language.extra.libdevice import rsqrt as _rsqrt
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import rsqrt as _rsqrt
    except ImportError:
        from triton.language.math import rsqrt as _rsqrt


@triton.jit
def _qk_norm_rope_fwd_kernel(
    Q, q_token_stride,
    K, k_token_stride,
    Q_W, K_W,
    COS, SIN, cos_token_stride,
    Q_OUT, qo_token_stride,
    K_OUT, ko_token_stride,
    Q_RSTD, q_rstd_token_stride,
    K_RSTD, k_rstd_token_stride,
    N, S_COS,
    eps,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
    DO_NORM: tl.constexpr,
):
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

    # Load weight halves once per program (shared across all heads).
    if DO_NORM:
        q_w_l = tl.load(Q_W + dh, mask=dh_mask, other=0.0)
        q_w_r = tl.load(Q_W + dh + (hd // 2), mask=dh_mask, other=0.0)
        k_w_l = tl.load(K_W + dh, mask=dh_mask, other=0.0)
        k_w_r = tl.load(K_W + dh + (hd // 2), mask=dh_mask, other=0.0)

    h_q = tl.arange(0, pad_n_qh)[:, None]
    d_col = dh[None, :]
    q_mask = (h_q < n_qh) & (d_col < (hd // 2))

    h_k = tl.arange(0, pad_n_kh)[:, None]
    k_mask = (h_k < n_kh) & (d_col < (hd // 2))

    # --- Q ---
    q_base = Q + pid * q_token_stride
    qo_base = Q_OUT + pid * qo_token_stride
    q_l_off = h_q * hd + d_col
    q_r_off = q_l_off + (hd // 2)

    q_l = tl.load(q_base + q_l_off, mask=q_mask, other=0.0)
    q_r = tl.load(q_base + q_r_off, mask=q_mask, other=0.0)
    q_dtype = q_l.dtype

    if DO_NORM:
        q_l_f32 = q_l.to(tl.float32)
        q_r_f32 = q_r.to(tl.float32)
        # Per-head RMS over full hd.
        sum_sq_q = tl.sum(q_l_f32 * q_l_f32, axis=1) + tl.sum(q_r_f32 * q_r_f32, axis=1)
        q_rstd = _rsqrt(sum_sq_q / hd + eps)  # (pad_n_qh,)
        tl.store(
            Q_RSTD + pid * q_rstd_token_stride + tl.arange(0, pad_n_qh),
            q_rstd,
            mask=tl.arange(0, pad_n_qh) < n_qh,
        )
        q_l_hat = (q_l_f32 * q_rstd[:, None]).to(q_dtype) * q_w_l[None, :]
        q_r_hat = (q_r_f32 * q_rstd[:, None]).to(q_dtype) * q_w_r[None, :]
    else:
        q_l_hat = q_l
        q_r_hat = q_r

    # RoPE: y_l = x_l*c − x_r*s, y_r = x_l*s + x_r*c
    q_l_hat_f = q_l_hat.to(tl.float32)
    q_r_hat_f = q_r_hat.to(tl.float32)
    q_out_l = q_l_hat_f * cos_row - q_r_hat_f * sin_row
    q_out_r = q_l_hat_f * sin_row + q_r_hat_f * cos_row
    tl.store(qo_base + q_l_off, q_out_l.to(q_dtype), mask=q_mask)
    tl.store(qo_base + q_r_off, q_out_r.to(q_dtype), mask=q_mask)

    # --- K ---
    k_base = K + pid * k_token_stride
    ko_base = K_OUT + pid * ko_token_stride
    k_l_off = h_k * hd + d_col
    k_r_off = k_l_off + (hd // 2)

    k_l = tl.load(k_base + k_l_off, mask=k_mask, other=0.0)
    k_r = tl.load(k_base + k_r_off, mask=k_mask, other=0.0)

    if DO_NORM:
        k_l_f32 = k_l.to(tl.float32)
        k_r_f32 = k_r.to(tl.float32)
        sum_sq_k = tl.sum(k_l_f32 * k_l_f32, axis=1) + tl.sum(k_r_f32 * k_r_f32, axis=1)
        k_rstd = _rsqrt(sum_sq_k / hd + eps)
        tl.store(
            K_RSTD + pid * k_rstd_token_stride + tl.arange(0, pad_n_kh),
            k_rstd,
            mask=tl.arange(0, pad_n_kh) < n_kh,
        )
        k_l_hat = (k_l_f32 * k_rstd[:, None]).to(q_dtype) * k_w_l[None, :]
        k_r_hat = (k_r_f32 * k_rstd[:, None]).to(q_dtype) * k_w_r[None, :]
    else:
        k_l_hat = k_l
        k_r_hat = k_r

    k_l_hat_f = k_l_hat.to(tl.float32)
    k_r_hat_f = k_r_hat.to(tl.float32)
    k_out_l = k_l_hat_f * cos_row - k_r_hat_f * sin_row
    k_out_r = k_l_hat_f * sin_row + k_r_hat_f * cos_row
    tl.store(ko_base + k_l_off, k_out_l.to(q_dtype), mask=k_mask)
    tl.store(ko_base + k_r_off, k_out_r.to(q_dtype), mask=k_mask)


@triton.jit
def _qk_norm_rope_bwd_kernel(
    # Incoming grads (grad_q_rope, grad_k_rope)
    dQ_IN, dq_in_token_stride,
    dK_IN, dk_in_token_stride,
    # Saved forward tensors
    Q, q_token_stride,
    K, k_token_stride,
    Q_W, K_W,
    COS, SIN, cos_token_stride,
    Q_RSTD, q_rstd_token_stride,
    K_RSTD, k_rstd_token_stride,
    # Outputs
    dQ, dq_token_stride,
    dK, dk_token_stride,
    dQ_W, dq_w_row_stride,  # (sm_count, hd) partials
    dK_W, dk_w_row_stride,
    # Dims
    N, S_COS,
    rows_per_program,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
    DO_NORM: tl.constexpr,
):
    """Backward through ``RoPE ∘ (weight × RMSNorm)``.

    Per-row (per-token × per-head) formulas:

      Let hd = head_dim, q_hat = q * rstd (pre-weight norm output),
          g = grad_q_rope, w = q_weight.

      Step 1 (rope inverse):
          gs_l = g_l * cos + g_r * sin   (gradient of weight*q_hat, left half)
          gs_r = -g_l * sin + g_r * cos  (right half)

      Step 2 (weight mul chain rule):
          m = gs * w   (in accum dtype, element-wise with broadcast)
          grad_q_weight = sum over tokens, heads of (gs * q_hat)

      Step 3 (RMS norm inverse):
          grad_q = rstd * (m - q_hat * mean(m * q_hat over hd))
    """
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    row_end = tl.minimum((block_id + 1) * rows_per_program, N)

    dh = tl.arange(0, pad_hd_half)
    dh_mask = dh < (hd // 2)

    # Load weights and init dW accumulators.
    if DO_NORM:
        q_w_l = tl.load(Q_W + dh, mask=dh_mask, other=0.0)
        q_w_r = tl.load(Q_W + dh + (hd // 2), mask=dh_mask, other=0.0)
        k_w_l = tl.load(K_W + dh, mask=dh_mask, other=0.0)
        k_w_r = tl.load(K_W + dh + (hd // 2), mask=dh_mask, other=0.0)
        dQ_W_l = tl.zeros((pad_hd_half,), dtype=tl.float32)
        dQ_W_r = tl.zeros((pad_hd_half,), dtype=tl.float32)
        dK_W_l = tl.zeros((pad_hd_half,), dtype=tl.float32)
        dK_W_r = tl.zeros((pad_hd_half,), dtype=tl.float32)

    h_q = tl.arange(0, pad_n_qh)[:, None]
    d_col = dh[None, :]
    q_mask = (h_q < n_qh) & (d_col < (hd // 2))

    h_k = tl.arange(0, pad_n_kh)[:, None]
    k_mask = (h_k < n_kh) & (d_col < (hd // 2))

    for pid in range(row_start, row_end):
        tok_cos = pid % S_COS
        c_base = COS + tok_cos * cos_token_stride
        s_base = SIN + tok_cos * cos_token_stride
        cos_row = tl.load(c_base + dh, mask=dh_mask, other=0.0)
        sin_row = tl.load(s_base + dh, mask=dh_mask, other=0.0)

        # ---- Q ----
        q_base = Q + pid * q_token_stride
        dq_in_base = dQ_IN + pid * dq_in_token_stride
        dq_base = dQ + pid * dq_token_stride
        q_l_off = h_q * hd + d_col
        q_r_off = q_l_off + (hd // 2)

        g_l = tl.load(dq_in_base + q_l_off, mask=q_mask, other=0.0).to(tl.float32)
        g_r = tl.load(dq_in_base + q_r_off, mask=q_mask, other=0.0).to(tl.float32)
        # Rope inverse
        gs_l = g_l * cos_row + g_r * sin_row
        gs_r = -g_l * sin_row + g_r * cos_row

        q_l = tl.load(q_base + q_l_off, mask=q_mask, other=0.0)
        q_r = tl.load(q_base + q_r_off, mask=q_mask, other=0.0)
        q_dtype = q_l.dtype

        if DO_NORM:
            rstd = tl.load(
                Q_RSTD + pid * q_rstd_token_stride + tl.arange(0, pad_n_qh),
                mask=tl.arange(0, pad_n_qh) < n_qh,
                other=0.0,
            )  # (pad_n_qh,)
            q_hat_l = q_l.to(tl.float32) * rstd[:, None]
            q_hat_r = q_r.to(tl.float32) * rstd[:, None]

            m_l = gs_l * q_w_l[None, :]
            m_r = gs_r * q_w_r[None, :]

            # mean(m * q_hat) over hd (summing both halves).
            dot = (tl.sum(m_l * q_hat_l, axis=1) + tl.sum(m_r * q_hat_r, axis=1)) / hd

            dq_l = rstd[:, None] * (m_l - q_hat_l * dot[:, None])
            dq_r = rstd[:, None] * (m_r - q_hat_r * dot[:, None])
            tl.store(dq_base + q_l_off, dq_l.to(q_dtype), mask=q_mask)
            tl.store(dq_base + q_r_off, dq_r.to(q_dtype), mask=q_mask)

            # dW partial: sum over heads of (gs * q_hat) (already cast to fp32).
            dQ_W_l += tl.sum(gs_l * q_hat_l, axis=0)
            dQ_W_r += tl.sum(gs_r * q_hat_r, axis=0)
        else:
            # No norm: grad_q = gs directly (after rope inverse).
            tl.store(dq_base + q_l_off, gs_l.to(q_dtype), mask=q_mask)
            tl.store(dq_base + q_r_off, gs_r.to(q_dtype), mask=q_mask)

        # ---- K ----
        k_base = K + pid * k_token_stride
        dk_in_base = dK_IN + pid * dk_in_token_stride
        dk_base = dK + pid * dk_token_stride
        k_l_off = h_k * hd + d_col
        k_r_off = k_l_off + (hd // 2)

        gk_l = tl.load(dk_in_base + k_l_off, mask=k_mask, other=0.0).to(tl.float32)
        gk_r = tl.load(dk_in_base + k_r_off, mask=k_mask, other=0.0).to(tl.float32)
        gsk_l = gk_l * cos_row + gk_r * sin_row
        gsk_r = -gk_l * sin_row + gk_r * cos_row

        k_l = tl.load(k_base + k_l_off, mask=k_mask, other=0.0)
        k_r = tl.load(k_base + k_r_off, mask=k_mask, other=0.0)

        if DO_NORM:
            k_rstd = tl.load(
                K_RSTD + pid * k_rstd_token_stride + tl.arange(0, pad_n_kh),
                mask=tl.arange(0, pad_n_kh) < n_kh,
                other=0.0,
            )
            k_hat_l = k_l.to(tl.float32) * k_rstd[:, None]
            k_hat_r = k_r.to(tl.float32) * k_rstd[:, None]

            mk_l = gsk_l * k_w_l[None, :]
            mk_r = gsk_r * k_w_r[None, :]
            dotk = (tl.sum(mk_l * k_hat_l, axis=1) + tl.sum(mk_r * k_hat_r, axis=1)) / hd

            dk_l = k_rstd[:, None] * (mk_l - k_hat_l * dotk[:, None])
            dk_r = k_rstd[:, None] * (mk_r - k_hat_r * dotk[:, None])
            tl.store(dk_base + k_l_off, dk_l.to(q_dtype), mask=k_mask)
            tl.store(dk_base + k_r_off, dk_r.to(q_dtype), mask=k_mask)

            dK_W_l += tl.sum(gsk_l * k_hat_l, axis=0)
            dK_W_r += tl.sum(gsk_r * k_hat_r, axis=0)
        else:
            tl.store(dk_base + k_l_off, gsk_l.to(q_dtype), mask=k_mask)
            tl.store(dk_base + k_r_off, gsk_r.to(q_dtype), mask=k_mask)

    if DO_NORM:
        # Store dW partials — left and right halves of the hd dim.
        tl.store(dQ_W + block_id * dq_w_row_stride + dh, dQ_W_l, mask=dh_mask)
        tl.store(dQ_W + block_id * dq_w_row_stride + dh + (hd // 2), dQ_W_r, mask=dh_mask)
        tl.store(dK_W + block_id * dk_w_row_stride + dh, dK_W_l, mask=dh_mask)
        tl.store(dK_W + block_id * dk_w_row_stride + dh + (hd // 2), dK_W_r, mask=dh_mask)


def _normalize_inputs(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Normalize (q, k) to 3D ``(N, H, D)`` and cos/sin to 2D ``(S_cos, D)``."""
    is_padded = q.dim() == 4
    if is_padded:
        B, H_q, S, D = q.shape
        H_k = k.shape[1]
        # (B, H, S, D) → contig (B, S, H, D) → view (B*S, H, D).
        q_flat = q.transpose(1, 2).contiguous().view(B * S, H_q, D)
        k_flat = k.transpose(1, 2).contiguous().view(B * S, H_k, D)
        N = B * S
    else:
        N, H_q, D = q.shape
        H_k = k.shape[1]
        q_flat = q.contiguous()
        k_flat = k.contiguous()

    cos2d = cos.reshape(-1, cos.shape[-1]).contiguous()
    sin2d = sin.reshape(-1, sin.shape[-1]).contiguous()
    return q_flat, k_flat, cos2d, sin2d, is_padded, N, H_q, H_k, D


def _launch_fwd(q_flat, k_flat, q_w, k_w, cos2d, sin2d, eps: float, do_norm: bool):
    N, H_q, D = q_flat.shape
    H_k = k_flat.shape[1]

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)
    if do_norm:
        q_rstd = torch.empty(N, H_q, device=q_flat.device, dtype=torch.float32)
        k_rstd = torch.empty(N, H_k, device=k_flat.device, dtype=torch.float32)
    else:
        # Dummy 1-element tensors just to satisfy the kernel signature.
        q_rstd = torch.empty(1, 1, device=q_flat.device, dtype=torch.float32)
        k_rstd = torch.empty(1, 1, device=k_flat.device, dtype=torch.float32)

    pad_hd_half = max(16, triton.next_power_of_2(D // 2))
    pad_n_qh = max(1, triton.next_power_of_2(H_q))
    pad_n_kh = max(1, triton.next_power_of_2(H_k))
    # Empty weight tensors when DO_NORM=False — kernel won't load them.
    if q_w is None:
        q_w = torch.empty(1, device=q_flat.device, dtype=q_flat.dtype)
    if k_w is None:
        k_w = torch.empty(1, device=k_flat.device, dtype=k_flat.dtype)

    _qk_norm_rope_fwd_kernel[(N,)](
        q_flat, q_flat.stride(0),
        k_flat, k_flat.stride(0),
        q_w, k_w,
        cos2d, sin2d, cos2d.stride(0),
        q_out, q_out.stride(0),
        k_out, k_out.stride(0),
        q_rstd, q_rstd.stride(0),
        k_rstd, k_rstd.stride(0),
        N, cos2d.shape[0],
        eps,
        n_qh=H_q, n_kh=H_k, hd=D,
        pad_n_qh=pad_n_qh, pad_n_kh=pad_n_kh, pad_hd_half=pad_hd_half,
        DO_NORM=do_norm,
        num_warps=4,
    )
    return q_out, k_out, q_rstd, k_rstd


def _launch_bwd(grad_q, grad_k, q_flat, k_flat, q_w, k_w, cos2d, sin2d, q_rstd, k_rstd, do_norm: bool):
    N, H_q, D = q_flat.shape
    H_k = k_flat.shape[1]

    # Make grad contig if not.
    if not grad_q.is_contiguous():
        grad_q = grad_q.contiguous()
    if not grad_k.is_contiguous():
        grad_k = grad_k.contiguous()

    dq = torch.empty_like(q_flat)
    dk = torch.empty_like(k_flat)

    sm_count = torch.cuda.get_device_properties(q_flat.device).multi_processor_count
    rows_per_program = math.ceil(N / sm_count)

    if do_norm:
        _dQW = torch.empty((sm_count, D), dtype=torch.float32, device=q_w.device)
        _dKW = torch.empty((sm_count, D), dtype=torch.float32, device=k_w.device)
    else:
        _dQW = torch.empty(1, 1, dtype=torch.float32, device=q_flat.device)
        _dKW = torch.empty(1, 1, dtype=torch.float32, device=k_flat.device)
    if q_w is None:
        q_w = torch.empty(1, device=q_flat.device, dtype=q_flat.dtype)
    if k_w is None:
        k_w = torch.empty(1, device=k_flat.device, dtype=k_flat.dtype)

    pad_hd_half = max(16, triton.next_power_of_2(D // 2))
    pad_n_qh = max(1, triton.next_power_of_2(H_q))
    pad_n_kh = max(1, triton.next_power_of_2(H_k))

    _qk_norm_rope_bwd_kernel[(sm_count,)](
        grad_q, grad_q.stride(0),
        grad_k, grad_k.stride(0),
        q_flat, q_flat.stride(0),
        k_flat, k_flat.stride(0),
        q_w, k_w,
        cos2d, sin2d, cos2d.stride(0),
        q_rstd, q_rstd.stride(0),
        k_rstd, k_rstd.stride(0),
        dq, dq.stride(0),
        dk, dk.stride(0),
        _dQW, _dQW.stride(0),
        _dKW, _dKW.stride(0),
        N, cos2d.shape[0],
        rows_per_program,
        n_qh=H_q, n_kh=H_k, hd=D,
        pad_n_qh=pad_n_qh, pad_n_kh=pad_n_kh, pad_hd_half=pad_hd_half,
        DO_NORM=do_norm,
        num_warps=4,
    )

    if do_norm:
        dqw = _dQW.sum(dim=0).to(q_w.dtype)
        dkw = _dKW.sum(dim=0).to(k_w.dtype)
    else:
        dqw = None
        dkw = None
    return dq, dk, dqw, dkw


class _FusedQKNormRoPEFn(torch.autograd.Function):
    """Fused QK-Norm + RoPE autograd Function."""

    @staticmethod
    def forward(ctx, q, k, q_weight, k_weight, cos, sin, eps, do_norm: bool):
        q_flat, k_flat, cos2d, sin2d, is_padded, N, H_q, H_k, D = _normalize_inputs(q, k, cos, sin)
        q_out_flat, k_out_flat, q_rstd, k_rstd = _launch_fwd(
            q_flat, k_flat, q_weight, k_weight, cos2d, sin2d, eps, do_norm
        )
        if do_norm:
            ctx.save_for_backward(q_flat, k_flat, q_weight, k_weight, cos2d, sin2d, q_rstd, k_rstd)
        else:
            ctx.save_for_backward(q_flat, k_flat, cos2d, sin2d)
        ctx.is_padded = is_padded
        ctx.shape_info = (H_q, H_k, D)
        ctx.do_norm = do_norm
        if is_padded:
            B, _H_q, S, _D = q.shape
            ctx.padded_shape = (B, S, H_q, H_k, D)
            q_out = q_out_flat.view(B, S, H_q, D).transpose(1, 2)
            k_out = k_out_flat.view(B, S, H_k, D).transpose(1, 2)
        else:
            q_out = q_out_flat
            k_out = k_out_flat
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        do_norm = ctx.do_norm
        if do_norm:
            q_flat, k_flat, q_weight, k_weight, cos2d, sin2d, q_rstd, k_rstd = ctx.saved_tensors
        else:
            q_flat, k_flat, cos2d, sin2d = ctx.saved_tensors
            q_weight = k_weight = None
            q_rstd = k_rstd = None

        # Normalize grad_q/grad_k to 3D (N, H, D).
        if ctx.is_padded:
            B, S, H_q, H_k, D = ctx.padded_shape
            grad_q_flat = grad_q.transpose(1, 2).contiguous().view(B * S, H_q, D)
            grad_k_flat = grad_k.transpose(1, 2).contiguous().view(B * S, H_k, D)
        else:
            grad_q_flat = grad_q.contiguous()
            grad_k_flat = grad_k.contiguous()

        if q_rstd is None:
            q_rstd = torch.empty(1, 1, device=q_flat.device, dtype=torch.float32)
            k_rstd = torch.empty(1, 1, device=k_flat.device, dtype=torch.float32)

        dq_flat, dk_flat, dqw, dkw = _launch_bwd(
            grad_q_flat, grad_k_flat,
            q_flat, k_flat,
            q_weight, k_weight,
            cos2d, sin2d,
            q_rstd, k_rstd,
            do_norm,
        )

        if ctx.is_padded:
            B, S, H_q, H_k, D = ctx.padded_shape
            dq = dq_flat.view(B, S, H_q, D).transpose(1, 2).contiguous()
            dk = dk_flat.view(B, S, H_k, D).transpose(1, 2).contiguous()
        else:
            dq = dq_flat
            dk = dk_flat

        return dq, dk, dqw, dkw, None, None, None, None


def fused_qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Single-kernel ``RoPE(weight * RMSNorm(q))`` — force Triton path (CUDA only).

    Args:
        q: ``(B, H_q, S, D)`` or ``(T, H_q, D)``.
        k: ``(B, H_k, S, D)`` or ``(T, H_k, D)``.
        q_weight, k_weight: ``(D,)`` per-head-dim scale. Gemma users pass
            ``1.0 + self.weight`` here.
        cos, sin: same layouts supported by ``lset.kernels.rope.apply_rotary_pos_emb``.
        eps: RMSNorm epsilon.
    """
    return _FusedQKNormRoPEFn.apply(q, k, q_weight, k_weight, cos, sin, eps, True)


def fused_rope_only(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rope-only path through the same kernel (``DO_NORM=False``)."""
    return _FusedQKNormRoPEFn.apply(q, k, None, None, cos, sin, 0.0, False)


def _eager_rms_norm_rope(
    q: Tensor, k: Tensor, q_w: Tensor, k_w: Tensor, cos: Tensor, sin: Tensor, eps: float,
) -> tuple[Tensor, Tensor]:
    """Pure-PyTorch reference for CPU fallback + correctness testing."""

    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rms_weight(x: Tensor, w: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (w * (xf * torch.rsqrt(var + eps))).to(dtype)

    q_hat = _rms_weight(q, q_w)
    k_hat = _rms_weight(k, k_w)
    q_out = q_hat * cos + _rotate_half(q_hat) * sin
    k_out = k_hat * cos + _rotate_half(k_hat) * sin
    return q_out, k_out


def qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Auto-dispatch wrapper — Triton path on CUDA, eager fallback on CPU or
    when ``LSET_DISABLE_FUSED_QK_NORM_ROPE=1`` is set (for A/B benchmarking)."""
    import os

    if q.is_cuda and os.environ.get("LSET_DISABLE_FUSED_QK_NORM_ROPE") != "1":
        return fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps)
    return _eager_rms_norm_rope(q, k, q_weight, k_weight, cos, sin, eps)
