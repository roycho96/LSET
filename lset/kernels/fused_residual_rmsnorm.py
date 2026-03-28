"""Fused Residual-Add + RMSNorm — single Triton kernel for the most frequent
back-to-back pattern in transformer blocks.

Current (2 kernels + 1 intermediate):
  hidden = residual + attn_out           # kernel 1
  hidden = rms_norm(hidden, weight)      # kernel 2

Fused (1 kernel):
  hidden, new_residual = fused_residual_rms_norm(residual, attn_out, weight, eps)
  # In one kernel:
  #   new_residual = residual + attn_out
  #   hidden = rms_norm(new_residual)    (x_hat = new_residual * rstd)

Weight multiply stays OUTSIDE the custom Function (same as Phase G2 RMSNorm)
to avoid AccumulateGrad fill_ overhead in GradCache.

Saves: 1 full (batch × seq × hidden) tensor write per call.
28 layers × 2 = 56 calls per forward pass.
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
def _residual_rms_norm_fwd_kernel(
    RESIDUAL,       # [N, D] residual input
    ATTN_OUT,       # [N, D] attention/MLP output
    Y,              # [N, D] normalized output (x_hat, WITHOUT weight)
    NEW_RESIDUAL,   # [N, D] residual + attn_out (saved for next layer)
    RSTD,           # [N] reciprocal std for backward
    N, D,
    stride_r, stride_a, stride_y, stride_nr,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    # Pass 1: compute new_residual = residual + attn_out, accumulate sum of squares
    sum_sq = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        r = tl.load(RESIDUAL + row * stride_r + offs_d, mask=mask, other=0.0)
        a = tl.load(ATTN_OUT + row * stride_a + offs_d, mask=mask, other=0.0)
        nr = (r + a).to(tl.float32)
        # Write new_residual
        tl.store(NEW_RESIDUAL + row * stride_nr + offs_d, nr.to(r.dtype), mask=mask)
        sum_sq += tl.sum(nr * nr)

    # RMS normalization
    mean_sq = sum_sq / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(RSTD + row, rstd)

    # Pass 2: normalize (read new_residual back, write x_hat)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0)
        y = nr.to(tl.float32) * rstd
        tl.store(Y + row * stride_y + offs_d, y.to(nr.dtype), mask=mask)


# =============================================================================
# Backward Kernel
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
def _residual_rms_norm_bwd_kernel(
    GRAD_Y,         # [N, D] upstream gradient (includes weight effect)
    NEW_RESIDUAL,   # [N, D] saved from forward
    RSTD,           # [N] reciprocal std from forward
    GRAD_INPUT,     # [N, D] gradient for BOTH residual and attn_out (same grad)
    N, D,
    stride_gy, stride_nr, stride_gi,
    BLOCK_D: tl.constexpr,
):
    """Backward for fused residual + RMSNorm.

    The add (residual + attn_out) has gradient 1 for both inputs.
    So grad_residual = grad_attn_out = d_rms_norm / d(new_residual).
    """
    row = tl.program_id(0)
    if row >= N:
        return

    rstd = tl.load(RSTD + row)

    # Pass 1: dot = sum(x_hat * grad_y) / D
    dot = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        x_hat = nr * rstd
        dot += tl.sum(x_hat * gy)
    dot = dot / D

    # Pass 2: grad = rstd * (gy - x_hat * dot)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        x_hat = nr * rstd
        gi = rstd * (gy - x_hat * dot)
        tl.store(GRAD_INPUT + row * stride_gi + offs_d, gi.to(gy.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _residual_rms_norm_forward(residual: Tensor, attn_out: Tensor, eps: float):
    orig_shape = residual.shape
    D = orig_shape[-1]
    residual_2d = residual.reshape(-1, D).contiguous()
    attn_out_2d = attn_out.reshape(-1, D).contiguous()
    N = residual_2d.shape[0]

    y = torch.empty_like(residual_2d)
    new_residual = torch.empty_like(residual_2d)
    rstd = torch.empty(N, device=residual.device, dtype=torch.float32)

    _residual_rms_norm_fwd_kernel[(N,)](
        residual_2d, attn_out_2d, y, new_residual, rstd,
        N, D,
        residual_2d.stride(0), attn_out_2d.stride(0),
        y.stride(0), new_residual.stride(0),
        eps,
    )
    return y.reshape(orig_shape), new_residual.reshape(orig_shape), rstd


def _residual_rms_norm_backward(grad_y: Tensor, new_residual: Tensor, rstd: Tensor):
    orig_shape = grad_y.shape
    D = orig_shape[-1]
    grad_y_2d = grad_y.reshape(-1, D).contiguous()
    nr_2d = new_residual.reshape(-1, D).contiguous()
    N = nr_2d.shape[0]

    grad_input = torch.empty_like(nr_2d)
    _residual_rms_norm_bwd_kernel[(N,)](
        grad_y_2d, nr_2d, rstd, grad_input,
        N, D,
        grad_y_2d.stride(0), nr_2d.stride(0), grad_input.stride(0),
    )
    return grad_input.reshape(orig_shape)


# =============================================================================
# Autograd Function
# =============================================================================

class _FusedResidualRMSNormFn(torch.autograd.Function):
    """Fused residual-add + RMSNorm.

    Returns (x_hat, new_residual) where x_hat = rms_norm(residual + attn_out).
    Weight multiply happens outside in Python for clean GradCache gradients.
    """

    @staticmethod
    def forward(ctx, residual: Tensor, attn_out: Tensor, eps: float):
        y, new_residual, rstd = _residual_rms_norm_forward(residual, attn_out, eps)
        ctx.save_for_backward(new_residual, rstd)
        return y, new_residual

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_new_residual: Tensor):
        new_residual, rstd = ctx.saved_tensors
        # grad through rms_norm
        grad_input = _residual_rms_norm_backward(grad_y, new_residual, rstd)
        # grad_new_residual passes through from downstream residual connections
        grad_input = grad_input + grad_new_residual
        # Both residual and attn_out get the same gradient (d/dx(a+b) = 1 for both)
        return grad_input, grad_input, None


# =============================================================================
# Public API
# =============================================================================

_FUSED_THRESHOLD = 256


def fused_residual_rms_norm(
    residual: Tensor, attn_out: Tensor, weight: Tensor, eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fused residual-add + RMSNorm — force fused path.

    Returns:
        (normed_output, new_residual) where:
        - normed_output = weight * rms_norm(residual + attn_out)
        - new_residual = residual + attn_out
    """
    x_hat, new_residual = _FusedResidualRMSNormFn.apply(residual, attn_out, eps)
    return weight * x_hat.to(residual.dtype), new_residual


def residual_rms_norm(
    residual: Tensor, attn_out: Tensor, weight: Tensor, eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Residual-add + RMSNorm with automatic Triton dispatch."""
    import os
    if (os.environ.get("LSET_DISABLE_FUSED_RESIDUAL_RMSNORM") != "1"
            and residual.is_cuda
            and residual.numel() // residual.shape[-1] >= _FUSED_THRESHOLD):
        return fused_residual_rms_norm(residual, attn_out, weight, eps)
    # Fallback: standard PyTorch
    new_residual = residual + attn_out
    input_dtype = new_residual.dtype
    x_f32 = new_residual.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype), new_residual
