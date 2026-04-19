"""Fused Residual-Add + RMSNorm — single-pass Triton kernel.

Techniques match ``rmsnorm.py`` (Liger-style single-pass row load + weight
fold inside kernel + per-SM partial dW reduction). In addition:

  - The ``residual + attn_out`` add happens in the same kernel, storing
    ``new_residual`` for both the backward graph and downstream consumers
    (the next block's MLP residual).
  - Backward returns ``(d_residual, d_attn_out, None)`` — both are equal
    (since ``d/dx(a + b) = 1`` for each input) but we clone the second
    slot to avoid AccumulateGrad aliasing if a caller hooks on both.
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

_MAX_FUSED_SIZE = 65536

_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float64: tl.float64,
}


def _calc_settings(n: int) -> tuple[int, int]:
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > _MAX_FUSED_SIZE:
        raise RuntimeError(f"Residual-RMSNorm feature dim {n} exceeds {_MAX_FUSED_SIZE}.")
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4
    return BLOCK_SIZE, num_warps


@triton.jit
def _residual_rms_norm_fwd_kernel(
    Y_ptr, Y_row_stride,
    R_ptr, R_row_stride,
    A_ptr, A_row_stride,
    NR_ptr, NR_row_stride,
    W_ptr,
    RSTD_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """new_residual = r + a; y = weight * rms_norm(new_residual)."""
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    r = tl.load(R_ptr + row_idx * R_row_stride + cols, mask=mask, other=0.0)
    a = tl.load(A_ptr + row_idx * A_row_stride + cols, mask=mask, other=0.0)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)

    # New residual (store in input dtype for downstream kernels)
    nr_f32 = r.to(tl.float32) + a.to(tl.float32)
    nr_dtype = r.dtype
    tl.store(NR_ptr + row_idx * NR_row_stride + cols, nr_f32.to(nr_dtype), mask=mask)

    mean_sq = tl.sum(nr_f32 * nr_f32, axis=0) / n_cols
    rstd = _rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx, rstd)

    y = (nr_f32 * rstd).to(nr_dtype) * w
    tl.store(Y_ptr + row_idx * Y_row_stride + cols, y, mask=mask)


@triton.jit
def _residual_rms_norm_bwd_kernel(
    dY_ptr, dY_row_stride,
    dR_ptr, dR_row_stride,
    NR_ptr, NR_row_stride,
    NR_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """d_new_residual via row-local formula; dW partial per-CTA."""
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    row_end = tl.minimum((block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row_idx in range(row_start, row_end):
        dy = tl.load(dY_ptr + row_idx * dY_row_stride + cols, mask=mask, other=0.0)
        nr = tl.load(NR_ptr + row_idx * NR_row_stride + cols, mask=mask, other=0.0)
        rstd = tl.load(RSTD_ptr + row_idx)

        nr_f32 = nr.to(tl.float32)
        m = (dy * w).to(tl.float32)
        dot_mx = tl.sum(m * nr_f32, axis=0)
        d_nr = rstd * m - (rstd * rstd * rstd / n_cols) * dot_mx * nr_f32

        dW_row += (dy * (nr_f32 * rstd).to(NR_dtype)).to(tl.float32)

        tl.store(dR_ptr + row_idx * dR_row_stride + cols, d_nr.to(NR_dtype), mask=mask)

    tl.store(dW_ptr + block_id * dW_row_stride + cols, dW_row, mask=mask)


def _fwd(r: Tensor, a: Tensor, w: Tensor, eps: float):
    shape = r.shape
    D = shape[-1]
    r_2d = r.reshape(-1, D)
    a_2d = a.reshape(-1, D)
    if not r_2d.is_contiguous():
        r_2d = r_2d.contiguous()
    if not a_2d.is_contiguous():
        a_2d = a_2d.contiguous()
    N = r_2d.shape[0]

    BLOCK_SIZE, num_warps = _calc_settings(D)
    y = torch.empty_like(r_2d)
    nr = torch.empty_like(r_2d)
    rstd = torch.empty(N, device=r.device, dtype=torch.float32)

    _residual_rms_norm_fwd_kernel[(N,)](
        y, y.stride(0),
        r_2d, r_2d.stride(0),
        a_2d, a_2d.stride(0),
        nr, nr.stride(0),
        w,
        rstd,
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.view(shape), nr.view(shape), rstd, BLOCK_SIZE, num_warps


def _bwd(grad_y: Tensor, nr_2d: Tensor, w: Tensor, rstd: Tensor, BLOCK_SIZE: int, num_warps: int):
    shape = grad_y.shape
    D = shape[-1]
    grad_y_2d = grad_y.reshape(-1, D)
    if not grad_y_2d.is_contiguous():
        grad_y_2d = grad_y_2d.contiguous()
    nr_2d = nr_2d.reshape(-1, D)
    if not nr_2d.is_contiguous():
        nr_2d = nr_2d.contiguous()
    N = grad_y_2d.shape[0]

    sm_count = torch.cuda.get_device_properties(nr_2d.device).multi_processor_count
    rows_per_program = math.ceil(N / sm_count)

    d_input = torch.empty_like(grad_y_2d)
    _dW = torch.empty((sm_count, D), dtype=torch.float32, device=w.device)

    _residual_rms_norm_bwd_kernel[(sm_count,)](
        grad_y_2d, grad_y_2d.stride(0),
        d_input, d_input.stride(0),
        nr_2d, nr_2d.stride(0),
        _TRITON_DTYPE[nr_2d.dtype],
        w,
        rstd,
        _dW, _dW.stride(0),
        N, D,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dW = _dW.sum(dim=0).to(w.dtype)
    return d_input.view(shape), dW


class _FusedResidualRMSNormFn(torch.autograd.Function):
    """Fused residual-add + weight-folded RMSNorm.

    Forward returns ``(weight * rms_norm(residual + attn_out), residual + attn_out)``.
    Backward returns ``(d_residual, d_attn_out, d_weight, None)`` where
    ``d_residual == d_attn_out`` (the add passes grad through unchanged).
    The second slot is a clone so AccumulateGrad on distinct params cannot
    alias.
    """

    @staticmethod
    def forward(ctx, residual: Tensor, attn_out: Tensor, weight: Tensor, eps: float):
        y, nr, rstd, BLOCK_SIZE, num_warps = _fwd(residual, attn_out, weight, eps)
        ctx.save_for_backward(nr, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y, nr

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_new_residual: Tensor):
        nr, weight, rstd = ctx.saved_tensors
        d_input, dw = _bwd(grad_y, nr, weight, rstd, ctx.BLOCK_SIZE, ctx.num_warps)
        # grad_new_residual flows through any downstream residual use; add it
        # to d_input (which is currently d/d(new_residual) from the norm path).
        d_input = d_input + grad_new_residual
        return d_input, d_input.clone(), dw, None


_FUSED_THRESHOLD = 256


def fused_residual_rms_norm(
    residual: Tensor,
    attn_out: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Force Triton path. Returns ``(weight * rms_norm(residual + attn_out), residual + attn_out)``."""
    return _FusedResidualRMSNormFn.apply(residual, attn_out, weight, eps)


def residual_rms_norm(
    residual: Tensor,
    attn_out: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Residual-add + RMSNorm with automatic Triton dispatch."""
    import os

    D = residual.shape[-1]
    if (
        os.environ.get("LSET_DISABLE_FUSED_RESIDUAL_RMSNORM") != "1"
        and residual.is_cuda
        and residual.numel() // D >= _FUSED_THRESHOLD
        and triton.next_power_of_2(D) <= _MAX_FUSED_SIZE
    ):
        return fused_residual_rms_norm(residual, attn_out, weight, eps)
    # Fallback
    new_residual = residual + attn_out
    input_dtype = new_residual.dtype
    x_f32 = new_residual.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype), new_residual
