"""Fused Pool + L2 Normalize kernels."""

import torch
import triton
import triton.language as tl

from torch import Tensor

# =============================================================================
# Fused Gather + Normalize (for last_token / cls)
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
def _gather_normalize_fwd_kernel(
    HIDDEN,  # [T, D] packed hidden states
    INDICES,  # [M] gather indices
    Y,  # [M, D] output (normalized)
    NORMS,  # [M] L2 norms (for backward)
    M,
    D,
    stride_h,
    stride_y,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    idx = tl.load(INDICES + row)

    # Pass 1: gather + sum of squares
    sum_sq = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(HIDDEN + idx * stride_h + offs_d, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32)

    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, eps)
    inv_norm = 1.0 / norm
    tl.store(NORMS + row, norm)

    # Pass 2: normalize + write
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(HIDDEN + idx * stride_h + offs_d, mask=mask, other=0.0)
        y = x.to(tl.float32) * inv_norm
        tl.store(Y + row * stride_y + offs_d, y.to(x.dtype), mask=mask)


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
def _gather_normalize_bwd_kernel(
    GRAD_Y,  # [M, D] upstream gradient
    Y,  # [M, D] normalized output from forward
    NORMS,  # [M] norms from forward
    GRAD_HIDDEN,  # [T, D] output gradient (scatter into)
    INDICES,  # [M] gather indices
    M,
    D,
    stride_gy,
    stride_y,
    stride_gh,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    idx = tl.load(INDICES + row)
    norm = tl.load(NORMS + row)
    inv_norm = 1.0 / norm

    # dot = <y, grad_y>
    dot = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(y * gy)

    # grad_x = (gy - y * dot) / norm
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        gx = (gy - y * dot) * inv_norm
        # Scatter (atomic add since indices may overlap in theory — but for
        # last_token/cls they don't, so this is safe non-atomic)
        tl.store(GRAD_HIDDEN + idx * stride_gh + offs_d, gx.to(gy.dtype), mask=mask)


# =============================================================================
# Fused Divide + Normalize (for mean pooling after scatter_add_)
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
def _div_normalize_fwd_kernel(
    X,  # [M, D] summed hidden states (from scatter_add_)
    LENGTHS,  # [M] sequence lengths
    Y,  # [M, D] output (divided + normalized)
    NORMS,  # [M] L2 norms after division (for backward)
    M,
    D,
    stride_x,
    stride_y,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    length = tl.load(LENGTHS + row).to(tl.float32)
    length = tl.maximum(length, eps)
    inv_len = 1.0 / length

    # Pass 1: divide by length + accumulate sum of squares
    sum_sq = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        meaned = x.to(tl.float32) * inv_len
        sum_sq += tl.sum(meaned * meaned)

    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, eps)
    inv_norm = 1.0 / norm
    tl.store(NORMS + row, norm)

    # Pass 2: divide + normalize
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X + row * stride_x + offs_d, mask=mask, other=0.0)
        meaned = x.to(tl.float32) * inv_len
        y = meaned * inv_norm
        tl.store(Y + row * stride_y + offs_d, y.to(x.dtype), mask=mask)


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
def _div_normalize_bwd_kernel(
    GRAD_Y,  # [M, D]
    Y,  # [M, D] forward output
    NORMS,  # [M] norms from forward
    LENGTHS,  # [M] sequence lengths
    GRAD_X,  # [M, D] output gradient
    M,
    D,
    stride_gy,
    stride_y,
    stride_gx,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    norm = tl.load(NORMS + row)
    inv_norm = 1.0 / norm
    length = tl.load(LENGTHS + row).to(tl.float32)
    length = tl.maximum(length, eps)
    inv_len = 1.0 / length

    dot = 0.0
    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(y * gy)

    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        y = tl.load(Y + row * stride_y + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(GRAD_Y + row * stride_gy + offs_d, mask=mask, other=0.0).to(tl.float32)
        # Chain rule: d/dx of (x/len / ||x/len||) = inv_len * (gy - y*dot) / norm
        gx = inv_len * inv_norm * (gy - y * dot)
        tl.store(GRAD_X + row * stride_gx + offs_d, gx.to(gy.dtype), mask=mask)


# =============================================================================
# Autograd Functions
# =============================================================================


class _FusedGatherNormalizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: Tensor, indices: Tensor, total_tokens: int, eps: float):
        M = indices.shape[0]
        D = hidden_states.shape[-1]
        hidden_2d = hidden_states.reshape(-1, D).contiguous()
        indices_long = indices.long().contiguous()

        y = torch.empty(M, D, dtype=hidden_states.dtype, device=hidden_states.device)
        norms = torch.empty(M, device=hidden_states.device, dtype=torch.float32)

        _gather_normalize_fwd_kernel[(M,)](
            hidden_2d,
            indices_long,
            y,
            norms,
            M,
            D,
            hidden_2d.stride(0),
            y.stride(0),
            eps,
        )
        ctx.save_for_backward(y, norms, indices_long)
        ctx.total_tokens = total_tokens
        ctx.D = D
        ctx.dtype = hidden_states.dtype
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        y, norms, indices = ctx.saved_tensors
        M, D = y.shape

        grad_hidden = torch.zeros(ctx.total_tokens, D, dtype=ctx.dtype, device=grad_y.device)
        _gather_normalize_bwd_kernel[(M,)](
            grad_y.contiguous(),
            y,
            norms,
            grad_hidden,
            indices,
            M,
            D,
            grad_y.stride(0),
            y.stride(0),
            grad_hidden.stride(0),
        )
        return grad_hidden, None, None, None


class _FusedDivNormalizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, summed: Tensor, lengths: Tensor, eps: float):
        M, D = summed.shape
        summed_c = summed.contiguous()
        lengths_f = lengths.float().contiguous()

        y = torch.empty_like(summed_c)
        norms = torch.empty(M, device=summed.device, dtype=torch.float32)

        _div_normalize_fwd_kernel[(M,)](
            summed_c,
            lengths_f,
            y,
            norms,
            M,
            D,
            summed_c.stride(0),
            y.stride(0),
            eps,
        )
        ctx.save_for_backward(y, norms, lengths_f)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        y, norms, lengths = ctx.saved_tensors
        M, D = y.shape

        grad_x = torch.empty_like(y)
        _div_normalize_bwd_kernel[(M,)](
            grad_y.contiguous(),
            y,
            norms,
            lengths,
            grad_x,
            M,
            D,
            grad_y.stride(0),
            y.stride(0),
            grad_x.stride(0),
            ctx.eps,
        )
        return grad_x, None, None


# =============================================================================
# Public API
# =============================================================================


def fused_pool_normalize(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    strategy: str = "last_token",
    eps: float = 1e-12,
) -> Tensor:
    """Fused packed pooling + L2 normalize."""
    num_seqs = cu_seqlens.shape[0] - 1
    T = hidden_states.shape[0]
    D = hidden_states.shape[-1]

    if strategy == "last_token":
        indices = (cu_seqlens[1:] - 1).long()
        return _FusedGatherNormalizeFn.apply(hidden_states, indices, T, eps)
    elif strategy == "cls":
        indices = cu_seqlens[:-1].long()
        return _FusedGatherNormalizeFn.apply(hidden_states, indices, T, eps)
    elif strategy == "mean":
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device),
            lengths,
        )
        summed = torch.zeros(num_seqs, D, dtype=hidden_states.dtype, device=hidden_states.device)
        summed.scatter_add_(0, seq_ids.unsqueeze(-1).expand_as(hidden_states), hidden_states)
        return _FusedDivNormalizeFn.apply(summed, lengths, eps)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")
