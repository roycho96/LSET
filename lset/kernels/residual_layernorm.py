"""Fused Residual-Add + LayerNorm — single-pass Triton kernel.

Matches ``layernorm.py`` (single-pass + weight/bias fold + per-SM partial
dW/dB) and ``residual_rmsnorm.py`` (fused residual add). Used by BERT /
XLM-RoBERTa encoders and the LayerNorm variant of the block-boundary
fusion.
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
        raise RuntimeError(f"Residual-LayerNorm feature dim {n} exceeds {_MAX_FUSED_SIZE}.")
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
def _residual_layer_norm_fwd_kernel(
    Y_ptr, Y_row_stride,
    R_ptr, R_row_stride,
    A_ptr, A_row_stride,
    NR_ptr, NR_row_stride,
    W_ptr,
    B_ptr,
    MEAN_ptr,
    RSTD_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """new_residual = r + a; y = w * layer_norm(new_residual) + b."""
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    r = tl.load(R_ptr + row_idx * R_row_stride + cols, mask=mask, other=0.0)
    a = tl.load(A_ptr + row_idx * A_row_stride + cols, mask=mask, other=0.0)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    nr_dtype = r.dtype
    nr_acc = r.to(ACCUM_DTYPE) + a.to(ACCUM_DTYPE)
    tl.store(NR_ptr + row_idx * NR_row_stride + cols, nr_acc.to(nr_dtype), mask=mask)

    mean = tl.sum(nr_acc, axis=0) / n_cols
    sum_sq = tl.sum(nr_acc * nr_acc, axis=0) / n_cols
    var = sum_sq - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = _rsqrt(var + eps)

    tl.store(MEAN_ptr + row_idx, mean)
    tl.store(RSTD_ptr + row_idx, rstd)

    x_hat = (nr_acc - mean) * rstd
    y = x_hat.to(nr_dtype) * w + b
    tl.store(Y_ptr + row_idx * Y_row_stride + cols, y, mask=mask)


@triton.jit
def _residual_layer_norm_bwd_kernel(
    dY_ptr, dY_row_stride,
    dR_ptr, dR_row_stride,
    NR_ptr, NR_row_stride,
    NR_dtype: tl.constexpr,
    W_ptr,
    MEAN_ptr,
    RSTD_ptr,
    dW_ptr, dW_row_stride,
    dB_ptr, dB_row_stride,
    n_rows, n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    row_end = tl.minimum((block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=ACCUM_DTYPE)
    dB_row = tl.zeros((BLOCK_SIZE,), dtype=ACCUM_DTYPE)

    for row_idx in range(row_start, row_end):
        dy = tl.load(dY_ptr + row_idx * dY_row_stride + cols, mask=mask, other=0.0)
        nr = tl.load(NR_ptr + row_idx * NR_row_stride + cols, mask=mask, other=0.0)
        mean = tl.load(MEAN_ptr + row_idx)
        rstd = tl.load(RSTD_ptr + row_idx)

        nr_acc = nr.to(ACCUM_DTYPE)
        x_hat = (nr_acc - mean) * rstd

        m = (dy * w).to(ACCUM_DTYPE)
        sum_m = tl.sum(m, axis=0) / n_cols
        sum_m_xhat = tl.sum(m * x_hat, axis=0) / n_cols
        dx = rstd * (m - sum_m - x_hat * sum_m_xhat)

        dW_row += (dy * x_hat.to(NR_dtype)).to(ACCUM_DTYPE)
        dB_row += dy.to(ACCUM_DTYPE)

        tl.store(dR_ptr + row_idx * dR_row_stride + cols, dx.to(NR_dtype), mask=mask)

    tl.store(dW_ptr + block_id * dW_row_stride + cols, dW_row, mask=mask)
    tl.store(dB_ptr + block_id * dB_row_stride + cols, dB_row, mask=mask)


def _fwd(r: Tensor, a: Tensor, w: Tensor, b: Tensor, eps: float):
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
    use_fp64 = r.dtype == torch.float64
    stats_dtype = torch.float64 if use_fp64 else torch.float32

    y = torch.empty_like(r_2d)
    nr = torch.empty_like(r_2d)
    mean = torch.empty(N, device=r.device, dtype=stats_dtype)
    rstd = torch.empty(N, device=r.device, dtype=stats_dtype)

    _residual_layer_norm_fwd_kernel[(N,)](
        y, y.stride(0),
        r_2d, r_2d.stride(0),
        a_2d, a_2d.stride(0),
        nr, nr.stride(0),
        w, b,
        mean, rstd,
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        USE_FP64=use_fp64,
    )
    return y.view(shape), nr.view(shape), mean, rstd, BLOCK_SIZE, num_warps


def _bwd(grad_y, nr_2d, w, mean, rstd, BLOCK_SIZE, num_warps):
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

    use_fp64 = nr_2d.dtype == torch.float64
    accum_dtype = torch.float64 if use_fp64 else torch.float32
    d_input = torch.empty_like(grad_y_2d)
    _dW = torch.empty((sm_count, D), dtype=accum_dtype, device=w.device)
    _dB = torch.empty((sm_count, D), dtype=accum_dtype, device=w.device)

    _residual_layer_norm_bwd_kernel[(sm_count,)](
        grad_y_2d, grad_y_2d.stride(0),
        d_input, d_input.stride(0),
        nr_2d, nr_2d.stride(0),
        _TRITON_DTYPE[nr_2d.dtype],
        w,
        mean, rstd,
        _dW, _dW.stride(0),
        _dB, _dB.stride(0),
        N, D,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        USE_FP64=use_fp64,
    )
    dw = _dW.sum(dim=0).to(w.dtype)
    db = _dB.sum(dim=0).to(w.dtype)
    return d_input.view(shape), dw, db


class _FusedResidualLayerNormFn(torch.autograd.Function):
    """Fused residual-add + weight/bias-folded LayerNorm."""

    @staticmethod
    def forward(ctx, residual, attn_out, weight, bias, eps):
        y, nr, mean, rstd, BLOCK_SIZE, num_warps = _fwd(residual, attn_out, weight, bias, eps)
        ctx.save_for_backward(nr, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y, nr

    @staticmethod
    def backward(ctx, grad_y, grad_new_residual):
        nr, weight, mean, rstd = ctx.saved_tensors
        d_input, dw, db = _bwd(grad_y, nr, weight, mean, rstd, ctx.BLOCK_SIZE, ctx.num_warps)
        d_input = d_input + grad_new_residual
        return d_input, d_input.clone(), dw, db, None


_FUSED_THRESHOLD = 256


def fused_residual_layer_norm(
    residual: Tensor,
    attn_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Force Triton path. Returns ``(w * layer_norm(r + a) + b, r + a)``."""
    return _FusedResidualLayerNormFn.apply(residual, attn_out, weight, bias, eps)


def residual_layer_norm(
    residual: Tensor,
    attn_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Residual-add + LayerNorm with automatic Triton dispatch."""
    import os

    D = residual.shape[-1]
    if (
        os.environ.get("LSET_DISABLE_FUSED_LAYERNORM") != "1"
        and residual.is_cuda
        and residual.numel() // D >= _FUSED_THRESHOLD
        and triton.next_power_of_2(D) <= _MAX_FUSED_SIZE
    ):
        return fused_residual_layer_norm(residual, attn_out, weight, bias, eps)
    # Fallback
    new_residual = residual + attn_out
    normed = torch.nn.functional.layer_norm(new_residual, weight.shape, weight, bias, eps)
    return normed, new_residual
