"""SM120 experimental RMSNorm — multi-row CTA exploration (opt-in).

Measured on RTX 5060 Ti (SM120, 36 SMs), multi-row does NOT consistently
beat the single-row main kernel in ``lset/kernels/rmsnorm.py``:

  - Small N (≤2048): 1.14–1.27x slower than main (launch overhead dominates
    with fewer CTAs).
  - Large N (≥8192 with D≥4096): 0.96–0.98x — marginally faster, within
    benchmark noise.

Kept as a reference / starting point for future experiments (e.g., fused
qk_norm + RoPE on this same shape). Prefer the main kernel for training.

Techniques on top of the portable ``lset/kernels/rmsnorm.py``:
  1. ``BLOCK_ROW`` rows per CTA — 2D tile ``(BLOCK_ROW, BLOCK_SIZE)``.
  2. Hardware rsqrt via ``triton.language.extra.libdevice.rsqrt`` (also now
     used in the main kernel — no longer an experimental-only win).
  3. Persistent backward grid ``(sm_count,)`` with row striping.
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
        raise RuntimeError(f"RMSNorm feature dim {n} exceeds {_MAX_FUSED_SIZE}.")
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
def _rms_norm_fwd_block_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """Multi-row RMSNorm forward. One CTA handles ``BLOCK_ROW`` rows."""
    row_base = tl.program_id(0).to(tl.int64) * BLOCK_ROW
    rows = row_base + tl.arange(0, BLOCK_ROW)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n_rows
    col_mask = cols < n_cols

    x_ptrs = X_ptr + rows[:, None] * X_row_stride + cols[None, :]
    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0)

    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    x_dtype = x.dtype
    x_f32 = x.to(tl.float32)

    mean_sq = tl.sum(x_f32 * x_f32, axis=1) / n_cols
    rstd = _rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + rows, rstd, mask=row_mask)

    y = (x_f32 * rstd[:, None]).to(x_dtype) * w[None, :]
    tl.store(
        Y_ptr + rows[:, None] * Y_row_stride + cols[None, :],
        y,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _rms_norm_bwd_block_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """Multi-row backward with persistent ``(sm_count,)`` grid."""
    pid = tl.program_id(0).to(tl.int64)
    num_sms = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < n_cols

    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0)
    dW_tile = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(pid * BLOCK_ROW, n_rows, num_sms * BLOCK_ROW):
        rows = start + tl.arange(0, BLOCK_ROW)
        row_mask = rows < n_rows

        dy = tl.load(
            dY_ptr + rows[:, None] * dY_row_stride + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            X_ptr + rows[:, None] * X_row_stride + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        rstd = tl.load(RSTD_ptr + rows, mask=row_mask, other=0.0)

        x_f32 = x.to(tl.float32)
        m = (dy * w[None, :]).to(tl.float32)
        dot_mx = tl.sum(m * x_f32, axis=1)
        dx = rstd[:, None] * m - ((rstd * rstd * rstd / n_cols) * dot_mx)[:, None] * x_f32

        # dW partial: column-axis reduction across the BLOCK_ROW rows held.
        dW_tile += tl.sum((dy * (x_f32 * rstd[:, None]).to(X_dtype)).to(tl.float32), axis=0)

        tl.store(
            dX_ptr + rows[:, None] * dX_row_stride + cols[None, :],
            dx.to(X_dtype),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    tl.store(dW_ptr + pid * dW_row_stride + cols, dW_tile, mask=col_mask)


def _pick_block_row(D: int) -> int:
    # Heuristic: smaller D → bigger row block to saturate SMs.
    if D <= 1024:
        return 16
    if D <= 2048:
        return 8
    if D <= 4096:
        return 4
    return 2


def _fwd(x: Tensor, w: Tensor, eps: float):
    shape = x.shape
    D = shape[-1]
    x_2d = x.reshape(-1, D)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    N = x_2d.shape[0]

    BLOCK_SIZE, num_warps = _calc_settings(D)
    BLOCK_ROW = _pick_block_row(D)

    y = torch.empty_like(x_2d)
    rstd = torch.empty(N, device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK_ROW),)
    _rms_norm_fwd_block_kernel[grid](
        y, y.stride(0),
        x_2d, x_2d.stride(0),
        w,
        rstd,
        N, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_ROW=BLOCK_ROW,
        num_warps=num_warps,
    )
    return y.view(shape), x_2d, rstd, BLOCK_SIZE, num_warps, BLOCK_ROW


def _bwd(grad_y, x_2d, w, rstd, BLOCK_SIZE, num_warps, BLOCK_ROW):
    shape = grad_y.shape
    D = shape[-1]
    grad_y_2d = grad_y.reshape(-1, D)
    if not grad_y_2d.is_contiguous():
        grad_y_2d = grad_y_2d.contiguous()
    N = grad_y_2d.shape[0]

    sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
    dx = torch.empty_like(grad_y_2d)
    _dW = torch.empty((sm_count, D), dtype=torch.float32, device=w.device)

    _rms_norm_bwd_block_kernel[(sm_count,)](
        grad_y_2d, grad_y_2d.stride(0),
        dx, dx.stride(0),
        x_2d, x_2d.stride(0),
        _TRITON_DTYPE[x_2d.dtype],
        w,
        rstd,
        _dW, _dW.stride(0),
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_ROW=BLOCK_ROW,
        num_warps=num_warps,
    )
    dW = _dW.sum(dim=0).to(w.dtype)
    return dx.view(shape), dW


class _FusedRMSNormSM120Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        y, x_2d, rstd, BLOCK_SIZE, num_warps, BLOCK_ROW = _fwd(x, weight, eps)
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.BLOCK_ROW = BLOCK_ROW
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_2d, weight, rstd = ctx.saved_tensors
        dx, dw = _bwd(grad_y, x_2d, weight, rstd, ctx.BLOCK_SIZE, ctx.num_warps, ctx.BLOCK_ROW)
        return dx, dw, None


def fused_rms_norm_sm120(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Multi-row RMSNorm for SM120-class GPUs — opt-in experimental."""
    return _FusedRMSNormSM120Fn.apply(x, weight, eps)
