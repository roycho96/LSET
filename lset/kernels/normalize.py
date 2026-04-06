"""
Fused L2 Normalize
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

# Threshold: below this row count, kernel launch overhead dominates.
# Original H100 value: 8192. Re-calibrated for RTX 5060 Ti (SM100): 1024.
# Crossover measured at N=1024 with 1.09x speedup (D=1024, bf16).
_FUSED_NORM_THRESHOLD = 1024


# =============================================================================
# Forward Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _l2_norm_fwd_kernel(
    X,              # [N, D] input
    Y,              # [N, D] output (normalized)
    Norms,          # [N] output: per-row L2 norms (for backward)
    N,              # number of rows
    D,              # number of columns (hidden dim)
    stride_x_n,     # X row stride
    stride_x_d,     # X col stride
    stride_y_n,     # Y row stride
    stride_y_d,     # Y col stride
    eps,            # epsilon for numerical stability
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    # Pass 1: sum of squares (fp32 accumulation)
    sum_sq = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x_n + offs_d * stride_x_d,
                     mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32)

    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, eps)
    inv_norm = 1.0 / norm

    # Save norm for backward
    tl.store(Norms + row, norm)

    # Pass 2: normalize and write
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x_n + offs_d * stride_x_d,
                     mask=mask, other=0.0)
        y = x.to(tl.float32) * inv_norm
        tl.store(Y + row * stride_y_n + offs_d * stride_y_d,
                 y.to(x.dtype), mask=mask)


# =============================================================================
# Backward Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _l2_norm_bwd_kernel(
    GradY,          # [N, D] upstream gradient
    Y,              # [N, D] forward output (normalized x)
    Norms,          # [N] per-row L2 norms from forward
    GradX,          # [N, D] output: input gradient
    N,              # number of rows
    D,              # number of columns
    stride_gy_n, stride_gy_d,
    stride_y_n, stride_y_d,
    stride_gx_n, stride_gx_d,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    norm = tl.load(Norms + row)
    inv_norm = 1.0 / norm

    # Pass 1: dot product <y, grad_y>
    dot = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y_n + offs_d * stride_y_d,
                     mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GradY + row * stride_gy_n + offs_d * stride_gy_d,
                      mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(y * gy)

    # Pass 2: grad_x = (grad_y - y * dot) / norm
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y_n + offs_d * stride_y_d,
                     mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GradY + row * stride_gy_n + offs_d * stride_gy_d,
                      mask=mask, other=0.0).to(tl.float32)
        gx = (gy - y * dot) * inv_norm
        tl.store(GradX + row * stride_gx_n + offs_d * stride_gx_d,
                 gx.to(gy.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _l2_norm_forward(x: Tensor, eps: float = 1e-12):
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    N, D = x.shape

    x_cont = x.contiguous()
    y = torch.empty_like(x_cont)
    norms = torch.empty(N, device=x.device, dtype=torch.float32)

    grid = (N,)
    _l2_norm_fwd_kernel[grid](
        x_cont, y, norms,
        N, D,
        x_cont.stride(0), x_cont.stride(1),
        y.stride(0), y.stride(1),
        eps,
    )
    return y, norms


def _l2_norm_backward(grad_y: Tensor, y: Tensor, norms: Tensor):
    N, D = y.shape

    grad_y_cont = grad_y.contiguous()
    y_cont = y.contiguous()
    grad_x = torch.empty_like(grad_y_cont)

    grid = (N,)
    _l2_norm_bwd_kernel[grid](
        grad_y_cont, y_cont, norms, grad_x,
        N, D,
        grad_y_cont.stride(0), grad_y_cont.stride(1),
        y_cont.stride(0), y_cont.stride(1),
        grad_x.stride(0), grad_x.stride(1),
    )
    return grad_x


# =============================================================================
# Autograd Function
# =============================================================================

class FusedL2Normalize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, eps: float = 1e-12):
        y, norms = _l2_norm_forward(x, eps)
        ctx.save_for_backward(y, norms)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        y, norms = ctx.saved_tensors
        grad_x = _l2_norm_backward(grad_y, y, norms)
        return grad_x, None


# =============================================================================
# Public API
# =============================================================================

def fused_l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Fused L2 normalize — drop-in for F.normalize(x, p=2, dim=1).

    Args:
        x: (N, D) input tensor (bf16, fp16, or fp32)
        eps: epsilon for numerical stability

    Returns:
        (N, D) L2 normalized tensor (same dtype as input)
    """
    return FusedL2Normalize.apply(x, eps)


def normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """L2 normalize with automatic Triton dispatch.

    Uses fused Triton kernel when N >= threshold and on CUDA.
    Falls back to F.normalize otherwise.
    """
    if x.is_cuda and x.dim() == 2 and x.shape[0] >= _FUSED_NORM_THRESHOLD:
        return fused_l2_normalize(x, eps)
    return F.normalize(x, p=2, dim=-1)
