"""Fused RMSNorm — single-pass Triton kernel."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from torch import Tensor

# Hardware rsqrt via libdevice — ~20% faster than 1.0 / tl.sqrt on Blackwell.
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
    """BLOCK_SIZE and num_warps — same schedule as Liger's ``calculate_settings``."""
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > _MAX_FUSED_SIZE:
        raise RuntimeError(f"RMSNorm feature dim {n} exceeds kernel limit ({_MAX_FUSED_SIZE}).")
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
def _rms_norm_fwd_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """y = (x / sqrt(mean(x²) + eps)) * w, row-wise; saves ``rstd`` for backward."""
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X_ptr + row_idx * X_row_stride + cols, mask=mask, other=0.0)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)

    x_dtype = x.dtype
    x_f32 = x.to(tl.float32)
    mean_sq = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    rstd = _rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx, rstd)

    y = (x_f32 * rstd).to(x_dtype) * w
    tl.store(Y_ptr + row_idx * Y_row_stride + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """dX computed row-locally; dW partial accumulated per-CTA."""
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    row_end = tl.minimum((block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row_idx in range(row_start, row_end):
        dy = tl.load(dY_ptr + row_idx * dY_row_stride + cols, mask=mask, other=0.0)
        x = tl.load(X_ptr + row_idx * X_row_stride + cols, mask=mask, other=0.0)
        rstd = tl.load(RSTD_ptr + row_idx)

        x_f32 = x.to(tl.float32)
        m = (dy * w).to(tl.float32)
        dot_mx = tl.sum(m * x_f32, axis=0)
        dx = rstd * m - (rstd * rstd * rstd / n_cols) * dot_mx * x_f32

        # dW partial: dy * (x * rstd) summed over rows held by this program.
        dW_row += (dy * (x_f32 * rstd).to(X_dtype)).to(tl.float32)

        tl.store(dX_ptr + row_idx * dX_row_stride + cols, dx.to(X_dtype), mask=mask)

    tl.store(dW_ptr + block_id * dW_row_stride + cols, dW_row, mask=mask)


def _rms_norm_forward(x: Tensor, w: Tensor, eps: float):
    shape = x.shape
    D = shape[-1]
    x_2d = x.reshape(-1, D)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    N = x_2d.shape[0]

    BLOCK_SIZE, num_warps = _calc_settings(D)
    y = torch.empty_like(x_2d)
    rstd = torch.empty(N, device=x.device, dtype=torch.float32)

    _rms_norm_fwd_kernel[(N,)](
        y, y.stride(0),
        x_2d, x_2d.stride(0),
        w,
        rstd,
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.view(shape), x_2d, rstd, BLOCK_SIZE, num_warps


def _rms_norm_backward(grad_y: Tensor, x_2d: Tensor, w: Tensor, rstd: Tensor, BLOCK_SIZE: int, num_warps: int):
    shape = grad_y.shape
    D = shape[-1]
    grad_y_2d = grad_y.reshape(-1, D)
    if not grad_y_2d.is_contiguous():
        grad_y_2d = grad_y_2d.contiguous()
    N = grad_y_2d.shape[0]

    sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
    rows_per_program = math.ceil(N / sm_count)

    dx = torch.empty_like(grad_y_2d)
    _dW = torch.empty((sm_count, D), dtype=torch.float32, device=w.device)

    _rms_norm_bwd_kernel[(sm_count,)](
        grad_y_2d, grad_y_2d.stride(0),
        dx, dx.stride(0),
        x_2d, x_2d.stride(0),
        _TRITON_DTYPE[x_2d.dtype],
        w,
        rstd,
        _dW, _dW.stride(0),
        N, D,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dW = _dW.sum(dim=0).to(w.dtype)
    return dx.view(shape), dW


class _FusedRMSNormFn(torch.autograd.Function):
    """Weight-folded RMSNorm; returns ``weight * rms_norm(x)`` directly."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float):
        y, x_2d, rstd, BLOCK_SIZE, num_warps = _rms_norm_forward(x, weight, eps)
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        dx, dw = _rms_norm_backward(grad_y, x_2d, weight, rstd, ctx.BLOCK_SIZE, ctx.num_warps)
        return dx, dw, None


_FUSED_RMS_THRESHOLD = 256


def fused_rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused RMSNorm — force Triton path. ``y = weight * rms_norm(x)``."""
    return _FusedRMSNormFn.apply(x, weight, eps)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm with automatic Triton dispatch."""
    import os

    D = x.shape[-1]
    if (
        os.environ.get("LSET_DISABLE_FUSED_RMSNORM") != "1"
        and x.is_cuda
        and x.numel() // D >= _FUSED_RMS_THRESHOLD
        and triton.next_power_of_2(D) <= _MAX_FUSED_SIZE
    ):
        return fused_rms_norm(x, weight, eps)
    # Fallback: standard PyTorch.
    input_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype)
