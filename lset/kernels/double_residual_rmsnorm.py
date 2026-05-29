"""Fused ``rms_norm(a)·w1 → +residual → rms_norm(result)·w2`` (Gemma pattern)."""

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
        raise RuntimeError(f"feature dim {n} exceeds {_MAX_FUSED_SIZE}")
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
def _double_residual_rms_norm_fwd_kernel(
    A_ptr, A_row_stride,
    R_ptr, R_row_stride,
    W1_ptr, W2_ptr,
    OUT_ptr, OUT_row_stride,
    NR_ptr, NR_row_stride,
    RSTDA_ptr, RSTDR_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-wise: z = rms_norm(a)·w1 ; nr = r + z ; out = rms_norm(nr)·w2."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    a = tl.load(A_ptr + row * A_row_stride + cols, mask=mask, other=0.0)
    r = tl.load(R_ptr + row * R_row_stride + cols, mask=mask, other=0.0)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)

    a_dtype = a.dtype
    a_f32 = a.to(tl.float32)

    # Norm 1 on a
    sum_sq_a = tl.sum(a_f32 * a_f32, axis=0) / n_cols
    rstd_a = _rsqrt(sum_sq_a + eps)
    tl.store(RSTDA_ptr + row, rstd_a)
    z = (a_f32 * rstd_a).to(a_dtype) * w1

    # Residual add (store new_residual)
    nr = (r.to(tl.float32) + z.to(tl.float32))
    tl.store(NR_ptr + row * NR_row_stride + cols, nr.to(a_dtype), mask=mask)

    # Norm 2 on nr
    sum_sq_nr = tl.sum(nr * nr, axis=0) / n_cols
    rstd_nr = _rsqrt(sum_sq_nr + eps)
    tl.store(RSTDR_ptr + row, rstd_nr)
    out = (nr * rstd_nr).to(a_dtype) * w2

    tl.store(OUT_ptr + row * OUT_row_stride + cols, out, mask=mask)


@triton.jit
def _double_residual_rms_norm_bwd_kernel(
    # Incoming grads
    GOUT_ptr, GOUT_row_stride,
    GNR_ptr, GNR_row_stride,
    # Saved forward state
    A_ptr, A_row_stride,
    NR_ptr, NR_row_stride,
    W1_ptr, W2_ptr,
    RSTDA_ptr, RSTDR_ptr,
    # Outputs
    DA_ptr, DA_row_stride,
    DR_ptr, DR_row_stride,
    # Per-CTA dW partials
    DW1_ptr, DW1_row_stride,
    DW2_ptr, DW2_row_stride,
    # Dims
    A_dtype: tl.constexpr,
    n_rows, n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward through norm2 → residual-add → norm1."""
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    row_end = tl.minimum((block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)
    dW1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dW2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row in range(row_start, row_end):
        gout = tl.load(GOUT_ptr + row * GOUT_row_stride + cols, mask=mask, other=0.0)
        gnr = tl.load(GNR_ptr + row * GNR_row_stride + cols, mask=mask, other=0.0)
        a = tl.load(A_ptr + row * A_row_stride + cols, mask=mask, other=0.0)
        nr = tl.load(NR_ptr + row * NR_row_stride + cols, mask=mask, other=0.0)
        rstd_a = tl.load(RSTDA_ptr + row)
        rstd_nr = tl.load(RSTDR_ptr + row)

        a_f32 = a.to(tl.float32)
        nr_f32 = nr.to(tl.float32)

        # Norm 2 backward on nr → d_nr_from_norm2
        m2 = (gout * w2).to(tl.float32)
        nr_hat = nr_f32 * rstd_nr
        dot2 = tl.sum(m2 * nr_hat, axis=0) / n_cols
        d_nr_n2 = rstd_nr * (m2 - nr_hat * dot2)

        # Combine with downstream residual grad
        d_nr = d_nr_n2 + gnr.to(tl.float32)

        # Residual-add: dz = d_nr ; dr = d_nr
        dz = d_nr

        # Norm 1 backward on a → d_a
        m1 = (dz * w1.to(tl.float32)).to(tl.float32)
        a_hat = a_f32 * rstd_a
        dot1 = tl.sum(m1 * a_hat, axis=0) / n_cols
        d_a = rstd_a * (m1 - a_hat * dot1)

        tl.store(DA_ptr + row * DA_row_stride + cols, d_a.to(A_dtype), mask=mask)
        tl.store(DR_ptr + row * DR_row_stride + cols, d_nr.to(A_dtype), mask=mask)

        dW1 += (dz * a_hat.to(w1.dtype)).to(tl.float32)
        dW2 += (gout * nr_hat.to(w2.dtype)).to(tl.float32)

    tl.store(DW1_ptr + block_id * DW1_row_stride + cols, dW1, mask=mask)
    tl.store(DW2_ptr + block_id * DW2_row_stride + cols, dW2, mask=mask)


def _fwd(a: Tensor, r: Tensor, w1: Tensor, w2: Tensor, eps: float):
    shape = a.shape
    D = shape[-1]
    a_2d = a.reshape(-1, D)
    r_2d = r.reshape(-1, D)
    if not a_2d.is_contiguous():
        a_2d = a_2d.contiguous()
    if not r_2d.is_contiguous():
        r_2d = r_2d.contiguous()
    N = a_2d.shape[0]

    BLOCK_SIZE, num_warps = _calc_settings(D)

    out = torch.empty_like(a_2d)
    nr = torch.empty_like(a_2d)
    rstd_a = torch.empty(N, device=a.device, dtype=torch.float32)
    rstd_r = torch.empty(N, device=a.device, dtype=torch.float32)

    _double_residual_rms_norm_fwd_kernel[(N,)](
        a_2d, a_2d.stride(0),
        r_2d, r_2d.stride(0),
        w1, w2,
        out, out.stride(0),
        nr, nr.stride(0),
        rstd_a, rstd_r,
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out.view(shape), nr.view(shape), a_2d, rstd_a, rstd_r, BLOCK_SIZE, num_warps


def _bwd(grad_out, grad_nr, a_2d, nr, w1, w2, rstd_a, rstd_r, BLOCK_SIZE, num_warps):
    shape = grad_out.shape
    D = shape[-1]
    gout_2d = grad_out.reshape(-1, D)
    if not gout_2d.is_contiguous():
        gout_2d = gout_2d.contiguous()
    gnr_2d = grad_nr.reshape(-1, D)
    if not gnr_2d.is_contiguous():
        gnr_2d = gnr_2d.contiguous()
    nr_2d = nr.reshape(-1, D)
    if not nr_2d.is_contiguous():
        nr_2d = nr_2d.contiguous()
    N = gout_2d.shape[0]

    sm_count = torch.cuda.get_device_properties(a_2d.device).multi_processor_count
    rows_per_program = math.ceil(N / sm_count)

    da = torch.empty_like(a_2d)
    dr = torch.empty_like(a_2d)
    _dW1 = torch.empty((sm_count, D), dtype=torch.float32, device=w1.device)
    _dW2 = torch.empty((sm_count, D), dtype=torch.float32, device=w2.device)

    _double_residual_rms_norm_bwd_kernel[(sm_count,)](
        gout_2d, gout_2d.stride(0),
        gnr_2d, gnr_2d.stride(0),
        a_2d, a_2d.stride(0),
        nr_2d, nr_2d.stride(0),
        w1, w2,
        rstd_a, rstd_r,
        da, da.stride(0),
        dr, dr.stride(0),
        _dW1, _dW1.stride(0),
        _dW2, _dW2.stride(0),
        _TRITON_DTYPE[a_2d.dtype],
        N, D,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dW1 = _dW1.sum(dim=0).to(w1.dtype)
    dW2 = _dW2.sum(dim=0).to(w2.dtype)
    return da.view(shape), dr.view(shape), dW1, dW2


class _FusedDoubleResidualRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, r: Tensor, w1: Tensor, w2: Tensor, eps: float):
        out, nr, a_2d, rstd_a, rstd_r, BLOCK_SIZE, num_warps = _fwd(a, r, w1, w2, eps)
        ctx.save_for_backward(a_2d, nr, w1, w2, rstd_a, rstd_r)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return out, nr

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_nr: Tensor):
        a_2d, nr, w1, w2, rstd_a, rstd_r = ctx.saved_tensors
        da, dr, dw1, dw2 = _bwd(
            grad_out, grad_nr, a_2d, nr, w1, w2, rstd_a, rstd_r,
            ctx.BLOCK_SIZE, ctx.num_warps,
        )
        return da, dr, dw1, dw2, None


def _eager_double_residual_rms_norm(a, residual, w1, w2, eps):
    """Pure-PyTorch reference used for CPU / disable env / correctness tests."""
    def rms_norm(x, w):
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (w * (xf * torch.rsqrt(var + eps))).to(dtype)

    z = rms_norm(a, w1)
    nr = residual + z
    out = rms_norm(nr, w2)
    return out, nr


def fused_double_residual_rms_norm(
    a: Tensor,
    residual: Tensor,
    w1: Tensor,
    w2: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fused ``rms_norm(a)·w1 → +residual → rms_norm(result)·w2``; returns (out, new_residual)."""
    import os

    if not a.is_cuda or os.environ.get("LSET_DISABLE_FUSED_DOUBLE_RMSNORM") == "1":
        return _eager_double_residual_rms_norm(a, residual, w1, w2, eps)
    return _FusedDoubleResidualRMSNormFn.apply(a, residual, w1, w2, eps)
