"""Fused Residual-Add + LayerNorm — single Triton kernel for BERT/encoder post-norm.

BERT post-norm pattern:
  hidden = LayerNorm(residual + attn_out)

Current (separate ops):
  new_residual = residual + attn_out     # kernel 1
  hidden = layer_norm(new_residual)      # kernel 2+

Fused (1 kernel):
  x_hat, new_residual = fused_residual_layer_norm(residual, attn_out, eps)
  hidden = weight * x_hat + bias  (applied in Python for GradCache compat)

Same design as FusedResidualRMSNorm but with LayerNorm (mean subtraction + variance).
Weight and bias applied OUTSIDE the custom Function.
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
def _residual_layer_norm_fwd_kernel(
    RESIDUAL,       # [N, D] residual input
    ATTN_OUT,       # [N, D] attention/MLP output
    Y,              # [N, D] normalized output (x_hat, WITHOUT weight/bias)
    NEW_RESIDUAL,   # [N, D] residual + attn_out
    MEAN,           # [N] row means for backward
    RSTD,           # [N] reciprocal std for backward
    N, D,
    stride_r, stride_a, stride_y, stride_nr,
    eps,
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    row = tl.program_id(0)
    if row >= N:
        return

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    # Pass 1: compute new_residual = residual + attn_out, accumulate sum for mean
    row_sum = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        r = tl.load(RESIDUAL + row * stride_r + offs_d, mask=mask, other=0.0)
        a = tl.load(ATTN_OUT + row * stride_a + offs_d, mask=mask, other=0.0)
        nr = (r + a).to(ACCUM_DTYPE)
        tl.store(NEW_RESIDUAL + row * stride_nr + offs_d, nr.to(r.dtype), mask=mask)
        row_sum += tl.sum(nr)

    mean = row_sum / D

    # Pass 2: compute variance
    var_sum = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0)
        diff = nr.to(ACCUM_DTYPE) - mean
        diff = tl.where(mask, diff, 0.0)
        var_sum += tl.sum(diff * diff)

    var = var_sum / D
    rstd = tl.math.rsqrt(var + eps)

    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)

    # Pass 3: normalize
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0)
        y = (nr.to(ACCUM_DTYPE) - mean) * rstd
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
def _residual_layer_norm_bwd_kernel(
    GRAD_Y,         # [N, D] upstream gradient (includes weight effect)
    NEW_RESIDUAL,   # [N, D] saved from forward
    MEAN,           # [N] row means from forward
    RSTD,           # [N] reciprocal std from forward
    GRAD_INPUT,     # [N, D] gradient for BOTH residual and attn_out
    N, D,
    stride_gy, stride_nr, stride_gi,
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """Backward for fused residual + LayerNorm.

    The add has gradient 1 for both inputs.
    dx = rstd * (gy - mean(gy) - x_hat * mean(x_hat * gy))
    """
    row = tl.program_id(0)
    if row >= N:
        return

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    mean = tl.load(MEAN + row).to(ACCUM_DTYPE)
    rstd = tl.load(RSTD + row).to(ACCUM_DTYPE)

    # Pass 1: compute mean(gy) and mean(x_hat * gy)
    sum_gy = tl.zeros([], dtype=ACCUM_DTYPE)
    sum_xhat_gy = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        x_hat = (nr - mean) * rstd
        sum_gy += tl.sum(gy)
        sum_xhat_gy += tl.sum(x_hat * gy)
    mean_gy = sum_gy / D
    mean_xhat_gy = sum_xhat_gy / D

    # Pass 2: dx = rstd * (gy - mean_gy - x_hat * mean_xhat_gy)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        nr = tl.load(NEW_RESIDUAL + row * stride_nr + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        x_hat = (nr - mean) * rstd
        gi = rstd * (gy - mean_gy - x_hat * mean_xhat_gy)
        tl.store(GRAD_INPUT + row * stride_gi + offs_d, gi.to(gy.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _residual_layer_norm_forward(residual: Tensor, attn_out: Tensor, eps: float):
    orig_shape = residual.shape
    D = orig_shape[-1]
    residual_2d = residual.reshape(-1, D).contiguous()
    attn_out_2d = attn_out.reshape(-1, D).contiguous()
    N = residual_2d.shape[0]

    y = torch.empty_like(residual_2d)
    new_residual = torch.empty_like(residual_2d)
    stats_dtype = torch.float32 if residual.dtype != torch.float64 else torch.float64
    mean = torch.empty(N, device=residual.device, dtype=stats_dtype)
    rstd = torch.empty(N, device=residual.device, dtype=stats_dtype)

    use_fp64 = residual.dtype == torch.float64
    _residual_layer_norm_fwd_kernel[(N,)](
        residual_2d, attn_out_2d, y, new_residual, mean, rstd,
        N, D,
        residual_2d.stride(0), attn_out_2d.stride(0),
        y.stride(0), new_residual.stride(0),
        eps,
        USE_FP64=use_fp64,
    )
    return y.reshape(orig_shape), new_residual.reshape(orig_shape), mean, rstd


def _residual_layer_norm_backward(
    grad_y: Tensor, new_residual: Tensor, mean: Tensor, rstd: Tensor,
):
    orig_shape = grad_y.shape
    D = orig_shape[-1]
    grad_y_2d = grad_y.reshape(-1, D).contiguous()
    nr_2d = new_residual.reshape(-1, D).contiguous()
    N = nr_2d.shape[0]

    grad_input = torch.empty_like(nr_2d)
    use_fp64 = grad_y.dtype == torch.float64
    _residual_layer_norm_bwd_kernel[(N,)](
        grad_y_2d, nr_2d, mean, rstd, grad_input,
        N, D,
        grad_y_2d.stride(0), nr_2d.stride(0), grad_input.stride(0),
        USE_FP64=use_fp64,
    )
    return grad_input.reshape(orig_shape)


# =============================================================================
# Autograd Function
# =============================================================================

class _FusedResidualLayerNormFn(torch.autograd.Function):
    """Fused residual-add + LayerNorm.

    Returns (x_hat, new_residual) where x_hat = layer_norm(residual + attn_out).
    Weight/bias multiply happens outside in Python for clean GradCache gradients.
    """

    @staticmethod
    def forward(ctx, residual: Tensor, attn_out: Tensor, eps: float):
        y, new_residual, mean, rstd = _residual_layer_norm_forward(
            residual, attn_out, eps,
        )
        ctx.save_for_backward(new_residual, mean, rstd)
        return y, new_residual

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_new_residual: Tensor):
        new_residual, mean, rstd = ctx.saved_tensors
        grad_input = _residual_layer_norm_backward(grad_y, new_residual, mean, rstd)
        # grad_new_residual passes through from downstream residual connections
        grad_input = grad_input + grad_new_residual
        # Both residual and attn_out get the same gradient (d/dx(a+b) = 1 for both)
        return grad_input, grad_input, None


# =============================================================================
# Public API
# =============================================================================

_FUSED_THRESHOLD = 256


def fused_residual_layer_norm(
    residual: Tensor, attn_out: Tensor,
    weight: Tensor, bias: Tensor, eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Fused residual-add + LayerNorm — force fused path.

    Returns:
        (normed_output, new_residual) where:
        - normed_output = weight * layer_norm(residual + attn_out) + bias
        - new_residual = residual + attn_out
    """
    x_hat, new_residual = _FusedResidualLayerNormFn.apply(residual, attn_out, eps)
    return weight * x_hat.to(residual.dtype) + bias, new_residual


def residual_layer_norm(
    residual: Tensor, attn_out: Tensor,
    weight: Tensor, bias: Tensor, eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Residual-add + LayerNorm with automatic Triton dispatch."""
    if (residual.is_cuda
            and residual.numel() // residual.shape[-1] >= _FUSED_THRESHOLD):
        return fused_residual_layer_norm(residual, attn_out, weight, bias, eps)
    # Fallback: standard PyTorch
    new_residual = residual + attn_out
    normed = torch.nn.functional.layer_norm(
        new_residual, weight.shape, weight, bias, eps,
    )
    return normed, new_residual
