"""Fused LayerNorm — single-pass Triton kernel."""

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
        raise RuntimeError(f"LayerNorm feature dim {n} exceeds {_MAX_FUSED_SIZE}.")
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
def _layer_norm_fwd_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    B_ptr,
    MEAN_ptr,
    RSTD_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """y = w * (x − mean) * rstd + b, row-wise."""
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    x = tl.load(X_ptr + row_idx * X_row_stride + cols, mask=mask, other=0.0)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    x_dtype = x.dtype
    x_acc = x.to(ACCUM_DTYPE)
    # E[X²] − E[X]² single-pass variance (biased, torch-compatible).
    mean = tl.sum(x_acc, axis=0) / n_cols
    sum_sq = tl.sum(x_acc * x_acc, axis=0) / n_cols
    var = sum_sq - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = _rsqrt(var + eps)

    tl.store(MEAN_ptr + row_idx, mean)
    tl.store(RSTD_ptr + row_idx, rstd)

    x_hat = (x_acc - mean) * rstd
    y = x_hat.to(x_dtype) * w + b
    tl.store(Y_ptr + row_idx * Y_row_stride + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    X_dtype: tl.constexpr,
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
    """dx via standard LN gradient; dW and dB partials per-CTA."""
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
        x = tl.load(X_ptr + row_idx * X_row_stride + cols, mask=mask, other=0.0)
        mean = tl.load(MEAN_ptr + row_idx)
        rstd = tl.load(RSTD_ptr + row_idx)

        x_acc = x.to(ACCUM_DTYPE)
        x_hat = (x_acc - mean) * rstd

        m = (dy * w).to(ACCUM_DTYPE)
        sum_m = tl.sum(m, axis=0) / n_cols
        sum_m_xhat = tl.sum(m * x_hat, axis=0) / n_cols
        dx = rstd * (m - sum_m - x_hat * sum_m_xhat)

        dW_row += (dy * x_hat.to(X_dtype)).to(ACCUM_DTYPE)
        dB_row += dy.to(ACCUM_DTYPE)

        tl.store(dX_ptr + row_idx * dX_row_stride + cols, dx.to(X_dtype), mask=mask)

    tl.store(dW_ptr + block_id * dW_row_stride + cols, dW_row, mask=mask)
    tl.store(dB_ptr + block_id * dB_row_stride + cols, dB_row, mask=mask)


def _fwd(x: Tensor, w: Tensor, b: Tensor, eps: float):
    shape = x.shape
    D = shape[-1]
    x_2d = x.reshape(-1, D)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    N = x_2d.shape[0]

    BLOCK_SIZE, num_warps = _calc_settings(D)
    use_fp64 = x.dtype == torch.float64
    stats_dtype = torch.float64 if use_fp64 else torch.float32
    y = torch.empty_like(x_2d)
    mean = torch.empty(N, device=x.device, dtype=stats_dtype)
    rstd = torch.empty(N, device=x.device, dtype=stats_dtype)

    _layer_norm_fwd_kernel[(N,)](
        y, y.stride(0),
        x_2d, x_2d.stride(0),
        w, b,
        mean, rstd,
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        USE_FP64=use_fp64,
    )
    return y.view(shape), x_2d, mean, rstd, BLOCK_SIZE, num_warps


def _bwd(grad_y, x_2d, w, mean, rstd, BLOCK_SIZE, num_warps):
    shape = grad_y.shape
    D = shape[-1]
    grad_y_2d = grad_y.reshape(-1, D)
    if not grad_y_2d.is_contiguous():
        grad_y_2d = grad_y_2d.contiguous()
    N = grad_y_2d.shape[0]

    sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
    rows_per_program = math.ceil(N / sm_count)

    dx = torch.empty_like(grad_y_2d)
    use_fp64 = x_2d.dtype == torch.float64
    accum_dtype = torch.float64 if use_fp64 else torch.float32
    _dW = torch.empty((sm_count, D), dtype=accum_dtype, device=w.device)
    _dB = torch.empty((sm_count, D), dtype=accum_dtype, device=w.device)

    _layer_norm_bwd_kernel[(sm_count,)](
        grad_y_2d, grad_y_2d.stride(0),
        dx, dx.stride(0),
        x_2d, x_2d.stride(0),
        _TRITON_DTYPE[x_2d.dtype],
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
    return dx.view(shape), dw, db


class _FusedLayerNormFn(torch.autograd.Function):
    """Weight/bias-folded LayerNorm; returns ``w * (x − μ) * rstd + b``."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, x_2d, mean, rstd, BLOCK_SIZE, num_warps = _fwd(x, weight, bias, eps)
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_2d, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _bwd(grad_y, x_2d, weight, mean, rstd, ctx.BLOCK_SIZE, ctx.num_warps)
        return dx, dw, db, None


_FUSED_LAYERNORM_THRESHOLD = 256


def fused_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Fused LayerNorm — force Triton path. ``y = w * layer_norm(x) + b``."""
    return _FusedLayerNormFn.apply(x, weight, bias, eps)


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm with automatic Triton dispatch."""
    import os

    D = x.shape[-1]
    if (
        os.environ.get("LSET_DISABLE_FUSED_LAYERNORM") != "1"
        and x.is_cuda
        and x.numel() // D >= _FUSED_LAYERNORM_THRESHOLD
        and triton.next_power_of_2(D) <= _MAX_FUSED_SIZE
    ):
        return fused_layer_norm(x, weight, bias, eps)
    # Fallback: cuDNN / aten
    return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
