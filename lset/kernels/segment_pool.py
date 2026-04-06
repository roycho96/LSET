"""Triton packed segment mean pooling kernel.

Replaces scatter_add_ + repeat_interleave with a single Triton kernel that
reads hidden[cu_seqlens[seg]:cu_seqlens[seg+1]], computes the mean, and
optionally L2-normalizes in one pass.

Eliminates:
- repeat_interleave allocation for segment_ids
- scatter_add_ kernel
- Separate division by lengths

One program per sequence. Inner loop over tokens in the segment.
"""

import torch
import triton
import triton.language as tl

from torch import Tensor

# =============================================================================
# Forward Kernels
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def _segment_mean_fwd_kernel(
    HIDDEN,  # [T, H] packed hidden states
    CU_SEQLENS,  # [M+1] cumulative sequence lengths
    OUTPUT,  # [M, H] output (mean per segment)
    H: tl.constexpr,
    stride_h,  # HIDDEN stride for row
    stride_o,  # OUTPUT stride for row
    BLOCK_H: tl.constexpr,
):
    seg = tl.program_id(0)
    start = tl.load(CU_SEQLENS + seg).to(tl.int64)
    end = tl.load(CU_SEQLENS + seg + 1).to(tl.int64)
    length = end - start

    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for t in range(start, end):
            vals = tl.load(HIDDEN + t * stride_h + h_offs, mask=h_mask, other=0.0)
            acc += vals.to(tl.float32)

        mean = acc / length.to(tl.float32)
        tl.store(OUTPUT + seg * stride_o + h_offs, mean.to(OUTPUT.dtype.element_ty), mask=h_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def _segment_mean_normalize_fwd_kernel(
    HIDDEN,  # [T, H]
    CU_SEQLENS,  # [M+1]
    OUTPUT,  # [M, H] output (mean + L2-normalized)
    NORMS,  # [M] L2 norms (for backward)
    H: tl.constexpr,
    stride_h,
    stride_o,
    eps,
    BLOCK_H: tl.constexpr,
):
    seg = tl.program_id(0)
    start = tl.load(CU_SEQLENS + seg).to(tl.int64)
    end = tl.load(CU_SEQLENS + seg + 1).to(tl.int64)
    length = end - start

    # Pass 1: accumulate sum and compute mean, also accumulate norm
    sum_sq = 0.0
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for t in range(start, end):
            vals = tl.load(HIDDEN + t * stride_h + h_offs, mask=h_mask, other=0.0)
            acc += vals.to(tl.float32)

        mean = acc / length.to(tl.float32)
        sum_sq += tl.sum(mean * mean)

        # Store mean temporarily in output (overwritten in pass 2)
        tl.store(OUTPUT + seg * stride_o + h_offs, mean.to(OUTPUT.dtype.element_ty), mask=h_mask)

    # Compute L2 norm
    norm = tl.sqrt(sum_sq + eps)
    tl.store(NORMS + seg, norm)
    inv_norm = 1.0 / norm

    # Pass 2: normalize
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        mean = tl.load(OUTPUT + seg * stride_o + h_offs, mask=h_mask, other=0.0)
        normed = mean.to(tl.float32) * inv_norm
        tl.store(OUTPUT + seg * stride_o + h_offs, normed.to(OUTPUT.dtype.element_ty), mask=h_mask)


# =============================================================================
# Backward Kernel
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def _segment_mean_bwd_kernel(
    GRAD_OUTPUT,  # [M, H] upstream gradient
    CU_SEQLENS,  # [M+1]
    GRAD_HIDDEN,  # [T, H] gradient for hidden states
    H: tl.constexpr,
    stride_go,
    stride_gh,
    BLOCK_H: tl.constexpr,
):
    """Backward: grad for each token = grad_output[seg] / length."""
    seg = tl.program_id(0)
    start = tl.load(CU_SEQLENS + seg).to(tl.int64)
    end = tl.load(CU_SEQLENS + seg + 1).to(tl.int64)
    length = end - start
    inv_len = 1.0 / length.to(tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        go = tl.load(GRAD_OUTPUT + seg * stride_go + h_offs, mask=h_mask, other=0.0)
        grad_val = go.to(tl.float32) * inv_len

        for t in range(start, end):
            tl.store(GRAD_HIDDEN + t * stride_gh + h_offs, grad_val.to(GRAD_HIDDEN.dtype.element_ty), mask=h_mask)


# =============================================================================
# Autograd Functions
# =============================================================================


class _TritonSegmentMeanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens):
        M = cu_seqlens.shape[0] - 1
        H = hidden_states.shape[-1]
        output = torch.empty(M, H, dtype=hidden_states.dtype, device=hidden_states.device)
        _segment_mean_fwd_kernel[(M,)](
            hidden_states,
            cu_seqlens,
            output,
            H,
            hidden_states.stride(0),
            output.stride(0),
        )
        ctx.save_for_backward(cu_seqlens)
        ctx.T = hidden_states.shape[0]
        ctx.H = H
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        M = cu_seqlens.shape[0] - 1
        grad_hidden = torch.empty(ctx.T, ctx.H, dtype=grad_output.dtype, device=grad_output.device)
        _segment_mean_bwd_kernel[(M,)](
            grad_output.contiguous(),
            cu_seqlens,
            grad_hidden,
            ctx.H,
            grad_output.stride(0),
            grad_hidden.stride(0),
        )
        return grad_hidden, None


class _TritonSegmentMeanNormalizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, eps):
        M = cu_seqlens.shape[0] - 1
        H = hidden_states.shape[-1]
        output = torch.empty(M, H, dtype=hidden_states.dtype, device=hidden_states.device)
        norms = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        _segment_mean_normalize_fwd_kernel[(M,)](
            hidden_states,
            cu_seqlens,
            output,
            norms,
            H,
            hidden_states.stride(0),
            output.stride(0),
            eps,
        )
        ctx.save_for_backward(output, norms, cu_seqlens)
        ctx.T = hidden_states.shape[0]
        ctx.H = H
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, norms, cu_seqlens = ctx.saved_tensors
        M = cu_seqlens.shape[0] - 1

        # Grad through L2 normalize: d/dx (x/||x||) = (I - x*x^T/||x||^2) / ||x||
        # Simplified: grad_mean = (grad_output - output * dot(grad_output, output)) / norm
        dot = (grad_output * output).sum(dim=-1, keepdim=True)
        grad_mean = (grad_output - output * dot) / norms.unsqueeze(-1)

        # Grad through mean: each token gets grad_mean / length
        grad_hidden = torch.empty(ctx.T, ctx.H, dtype=grad_output.dtype, device=grad_output.device)
        _segment_mean_bwd_kernel[(M,)](
            grad_mean.contiguous(),
            cu_seqlens,
            grad_hidden,
            ctx.H,
            grad_mean.stride(0),
            grad_hidden.stride(0),
        )
        return grad_hidden, None, None


# =============================================================================
# Public API
# =============================================================================


def triton_segment_mean_pool(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Triton packed segment mean pooling.

    Args:
        hidden_states: (T, H) packed hidden states
        cu_seqlens: (M+1,) cumulative sequence lengths, int32
        normalize: Whether to L2-normalize output
        eps: epsilon for normalization

    Returns:
        (M, H) mean-pooled (and optionally normalized) embeddings
    """
    if normalize:
        return _TritonSegmentMeanNormalizeFn.apply(hidden_states, cu_seqlens, eps)
    else:
        return _TritonSegmentMeanFn.apply(hidden_states, cu_seqlens)
