"""Fused LayerNorm — single Triton kernel for encoder models (BERT, XLM-RoBERTa).

Standard LayerNorm:
  x_f32 = x.float()
  mean = x_f32.mean(-1)
  var = ((x_f32 - mean) ** 2).mean(-1)
  x_hat = (x_f32 - mean) / sqrt(var + eps)
  out = weight * x_hat + bias
  → ~8 kernel launches

Fused:
  One kernel: read x → compute mean, var → normalize → write x_hat.
  Weight and bias applied via standard PyTorch ops OUTSIDE the custom Function
  to avoid AccumulateGrad fill_ overhead in GradCache chunked backward.

Design follows FusedRMSNorm pattern exactly:
  kernel computes x_hat = (x - mean) * rstd
  Python does: weight * x_hat + bias
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
def _layer_norm_fwd_kernel(
    X,  # [N, D] input
    Y,  # [N, D] output (normalized x_hat, WITHOUT weight/bias)
    Mean,  # [N] row means (for backward)
    Rstd,  # [N] reciprocal std (for backward)
    N,  # number of rows
    D,  # number of columns
    stride_x,  # X row stride
    stride_y,  # Y row stride
    eps,  # epsilon
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    row = tl.program_id(0)
    if row >= N:
        return

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    # Pass 1: compute mean
    row_sum = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        row_sum += tl.sum(x.to(ACCUM_DTYPE))

    mean = row_sum / D

    # Pass 2: compute variance
    var_sum = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        diff = x.to(ACCUM_DTYPE) - mean
        diff = tl.where(mask, diff, 0.0)
        var_sum += tl.sum(diff * diff)

    var = var_sum / D
    rstd = tl.math.rsqrt(var + eps)

    # Save for backward
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Pass 3: normalize (no weight/bias — applied outside in Python)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        y = (x.to(ACCUM_DTYPE) - mean) * rstd
        tl.store(Y + row * stride_y + offs_d, y.to(x.dtype), mask=mask)


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
def _layer_norm_bwd_kernel(
    GradY,  # [N, D] upstream gradient (includes weight effect from chain rule)
    X,  # [N, D] input from forward
    Mean,  # [N] row means from forward
    Rstd,  # [N] reciprocal std from forward
    GradX,  # [N, D] output: input gradient
    N,
    D,
    stride_gy,
    stride_x,
    stride_gx,
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """Backward for LayerNorm.

    Forward: y = (x - mean) * rstd
    Backward: dx = rstd * (gy - mean(gy) - x_hat * mean(x_hat * gy))
    where x_hat = (x - mean) * rstd
    """
    row = tl.program_id(0)
    if row >= N:
        return

    ACCUM_DTYPE: tl.constexpr = tl.float64 if USE_FP64 else tl.float32

    mean = tl.load(Mean + row).to(ACCUM_DTYPE)
    rstd = tl.load(Rstd + row).to(ACCUM_DTYPE)

    # Pass 1: compute mean(gy) and mean(x_hat * gy)
    sum_gy = tl.zeros([], dtype=ACCUM_DTYPE)
    sum_xhat_gy = tl.zeros([], dtype=ACCUM_DTYPE)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        gy = tl.load(GradY + row * stride_gy + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        x_hat = (x - mean) * rstd
        sum_gy += tl.sum(gy)
        sum_xhat_gy += tl.sum(x_hat * gy)
    mean_gy = sum_gy / D
    mean_xhat_gy = sum_xhat_gy / D

    # Pass 2: dx = rstd * (gy - mean_gy - x_hat * mean_xhat_gy)
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        gy = tl.load(GradY + row * stride_gy + offs_d, mask=mask, other=0.0).to(ACCUM_DTYPE)
        x_hat = (x - mean) * rstd
        gx = rstd * (gy - mean_gy - x_hat * mean_xhat_gy)
        tl.store(GradX + row * stride_gx + offs_d, gx.to(gy.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================


def _layer_norm_forward(x: Tensor, eps: float):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    N, D = x_2d.shape

    y = torch.empty_like(x_2d)
    stats_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    mean = torch.empty(N, device=x.device, dtype=stats_dtype)
    rstd = torch.empty(N, device=x.device, dtype=stats_dtype)

    use_fp64 = x.dtype == torch.float64
    _layer_norm_fwd_kernel[(N,)](
        x_2d,
        y,
        mean,
        rstd,
        N,
        D,
        x_2d.stride(0),
        y.stride(0),
        eps,
        USE_FP64=use_fp64,
    )
    return y.reshape(orig_shape), mean, rstd


def _layer_norm_backward(grad_y: Tensor, x: Tensor, mean: Tensor, rstd: Tensor):
    orig_shape = grad_y.shape
    D = orig_shape[-1]
    grad_y_2d = grad_y.reshape(-1, D).contiguous()
    x_2d = x.reshape(-1, D).contiguous()
    N = x_2d.shape[0]

    grad_x = torch.empty_like(x_2d)
    use_fp64 = grad_y.dtype == torch.float64
    _layer_norm_bwd_kernel[(N,)](
        grad_y_2d,
        x_2d,
        mean,
        rstd,
        grad_x,
        N,
        D,
        grad_y_2d.stride(0),
        x_2d.stride(0),
        grad_x.stride(0),
        USE_FP64=use_fp64,
    )

    return grad_x.reshape(orig_shape)


# =============================================================================
# Autograd Function
# =============================================================================


class _FusedLayerNormFn(torch.autograd.Function):
    """Fused LayerNorm without weight/bias — they are applied via standard
    PyTorch ops so their gradients flow through normal autograd, avoiding
    AccumulateGrad fill_ overhead in GradCache chunked backward."""

    @staticmethod
    def forward(ctx, x: Tensor, eps: float):
        y, mean, rstd = _layer_norm_forward(x, eps)
        ctx.save_for_backward(x, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x, mean, rstd = ctx.saved_tensors
        grad_x = _layer_norm_backward(grad_y, x, mean, rstd)
        return grad_x, None


# =============================================================================
# Public API
# =============================================================================

_FUSED_LAYERNORM_THRESHOLD = 256


def fused_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Fused LayerNorm — force fused path.

    Normalization is done in a single Triton kernel. Weight scaling and bias
    addition are done via standard PyTorch ops so weight/bias gradients flow
    through normal autograd (avoids AccumulateGrad overhead in GradCache).
    """
    x_hat = _FusedLayerNormFn.apply(x, eps)
    return weight * x_hat.to(x.dtype) + bias


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm with automatic Triton dispatch."""
    import os

    if (
        os.environ.get("LSET_DISABLE_FUSED_LAYERNORM") != "1"
        and x.is_cuda
        and x.numel() // x.shape[-1] >= _FUSED_LAYERNORM_THRESHOLD
    ):
        return fused_layer_norm(x, weight, bias, eps)
    # Fallback: standard PyTorch
    return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
