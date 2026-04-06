"""Fused RMSNorm — single Triton kernel replacing 6 separate PyTorch ops.

Standard RMSNorm:
  x_f32 = x.float()                     # aten::copy_ (cast)
  variance = x_f32.pow(2).mean(-1)       # aten::pow, aten::mean
  x_f32 = x_f32 * rsqrt(variance + eps)  # aten::rsqrt, aten::mul
  out = weight * x_f32.to(input_dtype)    # aten::mul, aten::copy_

Fused:
  One kernel: read x → compute rms → normalize → scale by weight → write out.
  Forward: 1 read of x + 1 read of weight + 1 write = 2N*D + D reads, N*D write
  Backward: dx and dw computed in separate kernels

Per-layer savings: 6 kernel launches → 1 (forward), same for backward.
With 28 layers × 4 norms (input_ln, post_attn_ln, q_norm, k_norm) = 112 norm calls,
this eliminates ~560 kernel launches per forward pass.
"""

import torch
import triton
import triton.language as tl

from torch import Tensor

# =============================================================================
# Forward Kernel
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,  # [N, D] input
    Y,  # [N, D] output (normalized, WITHOUT weight scaling)
    Rstd,  # [N] reciprocal std (1/rms) for backward
    N,  # number of rows
    D,  # number of columns
    stride_x,  # X row stride
    stride_y,  # Y row stride
    eps,  # epsilon
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    # Accumulate sum of squares in fp32
    sum_sq = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32)

    # RMS = sqrt(mean(x^2))
    mean_sq = sum_sq / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Save rstd for backward
    tl.store(Rstd + row, rstd)

    # Normalize (no weight scaling — weight is applied via standard PyTorch mul
    # so its gradient flows through normal autograd, avoiding AccumulateGrad fill_
    # overhead in GradCache chunked backward)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y = x_f32 * rstd
        tl.store(Y + row * stride_y + offs_d, y.to(x.dtype), mask=mask)


# =============================================================================
# Backward Kernels
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _rms_norm_bwd_kernel(
    GradY,  # [N, D] upstream gradient (already includes weight effect)
    X,  # [N, D] input
    Rstd,  # [N] reciprocal std from forward
    GradX,  # [N, D] output: input gradient
    N,
    D,
    stride_gy,
    stride_x,
    stride_gx,
    BLOCK_D: tl.constexpr,
):
    """Compute dx for RMSNorm backward.

    Forward: y = x * rstd (no weight — weight applied via standard PyTorch mul)
    Backward: dx = rstd * (grad_y - x_hat * mean(x_hat * grad_y))
    where x_hat = x * rstd
    """
    row = tl.program_id(0)
    if row >= N:
        return

    rstd = tl.load(Rstd + row)

    # Pass 1: dot = sum(x_hat * grad_y) / D
    dot = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GradY + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        dot += tl.sum(x_hat * gy)
    dot = dot / D

    # Pass 2: dx = rstd * (gy - x_hat * dot)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GradY + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        gx = rstd * (gy - x_hat * dot)
        tl.store(GradX + row * stride_gx + offs_d, gx.to(gy.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================


def _rms_norm_forward(x: Tensor, eps: float):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    N, D = x_2d.shape

    y = torch.empty_like(x_2d)
    rstd = torch.empty(N, device=x.device, dtype=torch.float32)

    _rms_norm_fwd_kernel[(N,)](
        x_2d,
        y,
        rstd,
        N,
        D,
        x_2d.stride(0),
        y.stride(0),
        eps,
    )
    return y.reshape(orig_shape), rstd


def _rms_norm_backward(grad_y: Tensor, x: Tensor, rstd: Tensor):
    orig_shape = grad_y.shape
    D = orig_shape[-1]
    grad_y_2d = grad_y.reshape(-1, D).contiguous()
    x_2d = x.reshape(-1, D).contiguous()
    N = x_2d.shape[0]

    grad_x = torch.empty_like(x_2d)
    _rms_norm_bwd_kernel[(N,)](
        grad_y_2d,
        x_2d,
        rstd,
        grad_x,
        N,
        D,
        grad_y_2d.stride(0),
        x_2d.stride(0),
        grad_x.stride(0),
    )

    return grad_x.reshape(orig_shape)


# =============================================================================
# Autograd Function
# =============================================================================


class _FusedRMSNormFn(torch.autograd.Function):
    """Fused RMSNorm without weight — weight is applied via standard PyTorch mul
    so its gradient flows through normal autograd, avoiding AccumulateGrad
    fill_ overhead in GradCache chunked backward."""

    @staticmethod
    def forward(ctx, x: Tensor, eps: float):
        y, rstd = _rms_norm_forward(x, eps)
        ctx.save_for_backward(x, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x, rstd = ctx.saved_tensors
        grad_x = _rms_norm_backward(grad_y, x, rstd)
        return grad_x, None


# =============================================================================
# Public API
# =============================================================================

_FUSED_RMSNORM_THRESHOLD = 256


def fused_rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused RMSNorm — force fused path.

    Normalization is done in a single Triton kernel. Weight scaling is done
    via standard PyTorch multiply so weight gradients flow through normal
    autograd (avoids AccumulateGrad overhead in GradCache).
    """
    x_norm = _FusedRMSNormFn.apply(x, eps)
    return weight * x_norm.to(x.dtype)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm with automatic Triton dispatch."""
    if x.is_cuda and x.numel() // x.shape[-1] >= _FUSED_RMSNORM_THRESHOLD:
        return fused_rms_norm(x, weight, eps)
    # Fallback: standard PyTorch
    input_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype)
