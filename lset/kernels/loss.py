"""
Fused Dense Embedding Loss
"""

import math

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch import Tensor

# =============================================================================
# Constants
# =============================================================================

LOSS_MULTI = 0  # MP-NCE
LOSS_SOFT = 1  # Soft Label CE
LOSS_CROSS = 2  # Standard CE

_LOSS_TYPE_MAP = {"multi": LOSS_MULTI, "soft": LOSS_SOFT, "cross": LOSS_CROSS}

# LSE forward kernel mode (constexpr)
LSE_NEG_ONLY = 0  # labels == 0  (for MP-NCE)
LSE_VALID_ALL = 1  # labels >= 0  (for CE: pos+neg, excluding ignore)

_GN_THRESHOLD = 128
_GM_THRESHOLD = 128


# =============================================================================
# Threshold: should_use_fused()
# =============================================================================


def should_use_fused(
    num_queries: int,
    num_docs: int,
    loss_type: str = "multi",
) -> bool:
    """
    Determine whether to use the fused kernel (hierarchical Q + K threshold).

    Based on K/Q sweep benchmarks (H100, D=1024, bf16):
      - Q >= 2048: almost always beneficial (1.07~1.49x speed, 8~71% mem)
      - Q >= 1024 and K >= 4096: crossover point (~1.15x speed)
      - Q < 1024: dip zone (0.81~0.96x speed), memory always wins but speed loses
      - soft/cross have later crossover than multi (ref is simple matmul+logsumexp, cuBLAS advantaged)

    Difference from v5 fallback (K<512):
      - v5: K-only threshold -> fallback even at Q=4096,K=256 -> unnecessary score matrix
      - v6.1: Q-aware hierarchical -> fused used even for small K if Q is large enough

    Args:
        num_queries: Q (number of queries)
        num_docs: K (number of documents)
        loss_type: "multi", "soft", "cross"

    Returns:
        True to use fused kernel, False for reference (score matrix) path
    """
    if loss_type == "multi":
        # MULTI: beneficial regardless of K when Q>=2048, beneficial at K>=4096 when Q>=1024
        if num_queries >= 2048:
            return True
        elif num_queries >= 1024:
            return num_docs >= 4096
        return False
    else:
        # SOFT/CROSS: ref is simpler so fused crossover is later.
        # Fixed K>=4096 threshold (conservative)
        if num_queries >= 2048:
            return num_docs >= 4096
        return False


# =============================================================================
# Q Bucket (for autotune cache separation)
# =============================================================================


def _bucket_q(num_queries: int) -> int:
    """
    Bucket Q size into 3 ranges. Added to autotune key so that
    optimal configs are cached separately per Q size range.

    In the dip zone (Q<=512) small BLOCK_M (8/16) gets selected,
    while large Q (>1024) retains large BLOCK_M (16/32/64).

    Ranges:
      0: Q <= 512   (dip zone, fwd BM=16, dQ BM=8 preferred)
      1: Q <= 1024  (transition zone)
      2: Q > 1024   (large Q, existing configs preferred)

    CUBIN cache impact:
      Not tl.constexpr -> no increase in compiled variants, only best_config mapping separated.
      +2 buckets per kernel = only autotune cache entries increase (~15MB VRAM).
    """
    if num_queries <= 512:
        return 0
    elif num_queries <= 1024:
        return 1
    else:
        return 2


# =============================================================================
# Autotune configs
# =============================================================================


def _fwd_configs():
    """Forward LSE kernel configs. BM=16 added (v6.1): more CTAs for small Q."""
    return [
        # Existing BM=64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        # Existing BM=32
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        # v6.1 BM=16: 2x CTAs for Q<=512 (Q=512 -> 32 CTAs vs 16 CTAs with BM=32)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
    ]


def _bwd_dq_configs():
    """dQ backward kernel configs. BM=8 added (v6.1): 2x CTAs in dip zone."""
    return [
        # Existing BM=16 (added in v5, dQ accounts for ~50% of time)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        # Existing BM=64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        # Existing BM=32
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        # v6.1 BM=8: removed -- tl.dot minimum size M>=16 constraint makes BM=8 impossible
    ]


def _bwd_dk_configs():
    """dK backward kernel configs. Parallelized along K axis so q_bucket not needed, unchanged."""
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
    ]


# =============================================================================
# Forward Kernel: LogSumExp (LSE_MODE constexpr branching)
# =============================================================================


@triton.autotune(configs=_fwd_configs(), key=["hidden_dim", "q_bucket"])
@triton.jit
def _lse_fwd_kernel(
    # Tensor pointers
    Q,  # [num_queries, hidden_dim] query embeddings (already scaled)
    K,  # [num_docs, hidden_dim] document embeddings
    Labels,  # [num_queries, num_docs] int8 label matrix (-1=ignore, 0=neg, >0=pos)
    OutLSE,  # [num_queries] output: logsumexp value for each query
    # Dimension info
    num_queries,  # number of rows in Q (= number of queries)
    num_docs,  # number of rows in K (= number of documents)
    hidden_dim,  # embedding dimension (number of columns in Q, K)
    q_bucket,  # Q size bucket (0/1/2). For autotune cache separation. Not used inside kernel
    # stride: element distance to next row/col in the tensor
    # e.g.: memory location of Q[i,j] = Q_ptr + i * stride_q_m + j * stride_q_d
    stride_q_m,
    stride_q_d,  # Q (row, col) stride
    stride_k_n,
    stride_k_d,  # K (row, col) stride
    stride_l_m,
    stride_l_n,  # Labels (row, col) stride. stride_l_m = num_docs (skip one row = K elements)
    # tl.constexpr: compile-time constants. Changing values triggers recompilation to separate GPU binaries (CUBIN)
    BLOCK_M: tl.constexpr,  # Q-axis tile size (16/32/64). Number of queries processed by one CTA (thread block)
    BLOCK_N: tl.constexpr,  # K-axis tile size. Number of documents seen per inner loop iteration
    BLOCK_D: tl.constexpr,  # hidden_dim-axis tile size (128/256). Dot product computed in chunks of this size
    GROUP_N: tl.constexpr,  # K-axis tile group count. If 2, processes BLOCK_N*2 docs per loop iteration
    FP32_MODE: tl.constexpr,  # True if input is fp32. Controls tl.dot precision branching
    ALLOW_TF32: tl.constexpr,  # True to allow tf32 tensorcore (only meaningful when FP32_MODE)
    LSE_MODE: tl.constexpr,  # 0: neg only (labels==0 included, for MP-NCE)
    # 1: valid all (labels>=0 included, for CE)
    INT64_LABELS: tl.constexpr,  # True to use int64 for label pointer arithmetic.
    # Labels stride_l_m = num_docs, and row * num_docs can
    # overflow int32 max (2^31-1), reading wrong memory.
    # Set True when Q*K > 2^31.
):
    """
    Forward kernel computing logsumexp of selected scores for each query.

    Core idea:
      Normally, loss computation first builds score_matrix = Q @ K^T then computes logsumexp.
      This kernel avoids storing the score matrix by recomputing it tile-by-tile (small blocks)
      while incrementally computing logsumexp. This is the essence of "fused" -- memory savings.

    Online logsumexp algorithm:
      logsumexp(s1, s2, ..., sN) = max_s + log(sum(exp(si - max_s)))
      Since we can't see all scores at once, we process tiles one by one,
      incrementally updating the running max and running exp sum.
      If the new tile's max exceeds the current max, a correction factor is applied to the
      existing exp sum.

    Parallelization:
      GPU grid = (ceil(num_queries / BLOCK_M),)
      Each CTA (thread block) independently computes logsumexp for BLOCK_M queries.
      The K (document) direction is traversed sequentially in a for loop within each CTA.
    """
    # This CTA's index. The GPU runs multiple CTAs concurrently, each handling a different query block
    pid_m = tl.program_id(0)
    # Query indices assigned to this CTA. e.g.: pid_m=3, BLOCK_M=32 -> [96, 97, ..., 127]
    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    # Hidden dim direction offset. Used in the dot product inner loop
    offs_d = tl.arange(0, BLOCK_D)

    # GROUP_N technique: group K-direction tiles by GROUP_N.
    # This keeps the Q tile in registers while processing multiple K tiles,
    # reducing Q reload count. GROUP_N=2 processes BLOCK_N*2 docs at once.
    BLOCK_NG: tl.constexpr = BLOCK_N * GROUP_N
    # Doc offsets within current K tile group
    offs_ng = tl.arange(0, BLOCK_NG)

    # Online logsumexp state variables (one per query, BLOCK_M total)
    # m_i: running max of scores seen so far. Initialized to -inf
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")
    # lse_i: running sum(exp(score - running_max)). Initialized to 0
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # K-direction main loop: iterate over all documents in chunks of BLOCK_NG
    # e.g.: num_docs=1000, BLOCK_NG=128 -> 8 iterations (0, 128, 256, ..., 896)
    for start_n in range(0, num_docs, BLOCK_NG):
        # Document indices for the current tile
        cur_offs_n = start_n + offs_ng

        # [Load labels tile]
        # Read labels for the current (query, doc) block from Labels[query_idx, doc_idx]
        # label values: -1=ignore this (query,doc) pair, 0=negative pair, >0=positive pair
        #
        # Pointer calculation: Labels_ptr + query_row * stride_l_m + doc_col * stride_l_n
        # INT64_LABELS=True case: stride_l_m=num_docs is large, so query_row*num_docs
        # can exceed int32 range; cast offsets to int64 to compute
        if INT64_LABELS:
            label_ptrs = (
                Labels + offs_m[:, None].to(tl.int64) * stride_l_m + cur_offs_n[None, :].to(tl.int64) * stride_l_n
            )
        else:
            label_ptrs = Labels + offs_m[:, None] * stride_l_m + cur_offs_n[None, :] * stride_l_n
        # mask: for out-of-bounds indices, use the `other` value instead of reading memory
        # other=-1: out-of-bounds treated as ignore(-1), excluded from LSE
        labels_tile = tl.load(
            label_ptrs,
            mask=(offs_m[:, None] < num_queries) & (cur_offs_n[None, :] < num_docs),
            other=-1,
        )

        # [Create mask] Set True only for score positions to include in LSE
        if LSE_MODE == 0:
            # MP-NCE: only negative pair (label==0) scores go into logsumexp
            mask = labels_tile == 0
        else:
            # CE family: include both positive and negative, exclude only ignore(-1)
            mask = labels_tile >= 0

        # [Compute Q@K^T score tile]
        # Instead of building the full score matrix, compute only the small tile
        # for the current (query block, doc block). Result shape: [BLOCK_M, BLOCK_NG]
        #
        # If hidden_dim > BLOCK_D, split the dot product across multiple iterations
        # and accumulate partial sums. e.g.: hidden_dim=1024, BLOCK_D=256 -> 4 iterations
        qk = tl.zeros([BLOCK_M, BLOCK_NG], dtype=tl.float32)
        for start_d in range(0, hidden_dim, BLOCK_D):
            cur_offs_d = start_d + offs_d
            # Load [BLOCK_M, BLOCK_D] slice of Q for the current query block
            q_ptrs = Q + offs_m[:, None] * stride_q_m + cur_offs_d[None, :] * stride_q_d
            # Load [BLOCK_NG, BLOCK_D] slice of K for the current doc block
            k_ptrs = K + cur_offs_n[:, None] * stride_k_n + cur_offs_d[None, :] * stride_k_d
            # mask prevents out-of-bounds reads. other=0.0 has no effect on dot product
            q_tile = tl.load(
                q_ptrs, mask=(offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            k_tile = tl.load(
                k_ptrs, mask=(cur_offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            # tl.dot: matrix multiply using GPU tensor cores. [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_NG]
            # tl.trans(k_tile): transpose K from [BLOCK_NG, BLOCK_D] to [BLOCK_D, BLOCK_NG]
            if FP32_MODE and not ALLOW_TF32:
                # fp32 input with tf32 disallowed: use IEEE precision (most accurate but slowest)
                qk += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
            else:
                # bf16/fp16 input or tf32 allowed: use default tensor core precision
                qk += tl.dot(q_tile, tl.trans(k_tile))

        # [Apply mask] Set scores not included in LSE to -inf
        # exp(-inf) = 0, so they are naturally excluded from logsumexp
        qk = tl.where(mask, qk, float("-inf"))

        # [Update online logsumexp]
        # Per-query max score in the current tile
        tile_max = tl.max(qk, 1)
        # New global max = max of running max and current tile max
        m_ij = tl.maximum(m_i, tile_max)
        # exp(score - new_max): subtracting new_max prevents overflow
        # When max is large enough, exp values stay close to 0 for safe computation
        p = tl.exp(qk - m_ij[:, None])
        # Zero out positions that are out-of-bounds or masked (safety net)
        p = tl.where(mask & (cur_offs_n[None, :] < num_docs), p, 0.0)
        # Apply correction factor to the existing accumulated exp sum.
        # The old exp sum was computed relative to old_max, but now max changed to new_max,
        # so multiply by exp(old_max - new_max) to adjust the baseline.
        # e.g.: old_max=5, new_max=7 -> alpha=exp(5-7)=exp(-2) ~ 0.135
        alpha = tl.exp(m_i - m_ij)
        # When both are -inf, exp(-inf - (-inf)) = exp(NaN) prevention. Treat as 0
        alpha = tl.where(m_ij == float("-inf"), 0.0, alpha)
        # Update: (old exp sum * correction) + (current tile exp sum)
        lse_i = alpha * lse_i + tl.sum(p, 1)
        # Update running max
        m_i = m_ij

    # [Final logsumexp computation]
    # lse = max + log(sum_of_exp_values)
    # Mathematically: log(sum(exp(s_i))) = max + log(sum(exp(s_i - max)))
    out_lse = m_i + tl.log(lse_i)
    # Padding queries (due to BLOCK_M alignment exceeding actual query count) get -inf
    out_lse = tl.where(offs_m < num_queries, out_lse, float("-inf"))
    # Store result to OutLSE tensor
    tl.store(OutLSE + offs_m, out_lse, mask=offs_m < num_queries)


# =============================================================================
# Backward: dQ kernel (LOSS_TYPE branching)
# =============================================================================


@triton.autotune(configs=_bwd_dq_configs(), key=["hidden_dim", "q_bucket"], reset_to_zero=["dQ"])
@triton.jit
def _dq_bwd_kernel(
    # Tensor pointers
    Q,  # [num_queries, hidden_dim] query embeddings (same as forward)
    K,  # [num_docs, hidden_dim] document embeddings
    Labels,  # [num_queries, num_docs] int8 label matrix
    dQ,  # [num_queries, hidden_dim] output: query gradient (accumulated in fp32)
    RefLSE,  # [num_queries] logsumexp computed in forward
    #   multi: neg_lse (logsumexp of negative scores)
    #   soft/cross: all_lse (logsumexp of all valid scores)
    Aux,  # [num_queries] per-loss auxiliary value
    #   multi: sum_weights = per-query sum of sigma(score - neg_lse)
    #   soft/cross: label_sum = per-query sum of labels
    W,  # [num_queries] per-query loss weight (derived from softplus gradient)
    # Dimension info
    num_queries,
    num_docs,
    hidden_dim,
    q_bucket,  # Q size bucket (for autotune cache separation, not used in kernel)
    # stride
    stride_q_m,
    stride_q_d,  # Q, dQ (row, col) stride
    stride_k_n,
    stride_k_d,  # K (row, col) stride
    stride_l_m,
    stride_l_n,  # Labels (row, col) stride
    # tl.constexpr
    BLOCK_M: tl.constexpr,  # Q-axis tile size (16/32/64)
    BLOCK_N: tl.constexpr,  # K-axis tile size
    BLOCK_D: tl.constexpr,  # hidden_dim-axis tile size
    GROUP_N: tl.constexpr,  # K-axis tile group count
    FP32_MODE: tl.constexpr,  # True if input is fp32
    ALLOW_TF32: tl.constexpr,  # True to allow tf32 tensorcore
    CAST_DTYPE: tl.constexpr,  # Casting method before passing to tl.dot
    #   0: tf32 (inline asm to round fp32 to tf32)
    #   1: bf16 (cast grad_s to bf16, k_tile already bf16)
    #   2: fp16
    LOSS_TYPE: tl.constexpr,  # 0=multi(MP-NCE), 1=soft(Soft CE), 2=cross(Cross CE)
    # grad_s (gradient w.r.t. score) formula differs per loss type
    INT64_LABELS: tl.constexpr,  # True for int64 label pointer arithmetic (Q*K > 2^31)
):
    """
    dQ backward kernel: compute gradient for each query.

    Math:
      dQ[i] = sum_j grad_s[i,j] * K[j]
      i.e., multiply grad_s (gradient w.r.t. score) [Q,K] matrix with K [K,D] matrix to get dQ [Q,D].

    Process:
      1) Recompute Q@K^T scores tile-by-tile (same as forward, no score matrix stored)
      2) Compute grad_s using scores and RefLSE/Aux/W with per-loss-type formula
      3) Accumulate grad_s @ K_tile into dQ

    Parallelization:
      grid = (ceil(num_queries / BLOCK_M),)
      Each CTA computes dQ for BLOCK_M queries. K direction is sequential loop within CTA.
      reset_to_zero=["dQ"]: autotune benchmarks multiple configs, resets dQ to zero each time.
      Without this, previous benchmark results remain and gradients get inflated N-fold.
    """
    # PTX inline assembly for TF32 conversion. GPU instruction to round fp32 to tf32 precision.
    # Used before passing to tl.dot when CAST_DTYPE=0
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"

    # Query indices assigned to this CTA
    pid_m = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_d = tl.arange(0, BLOCK_D)

    # K-direction tile group (same technique as GROUP_N in forward kernel)
    BLOCK_NG: tl.constexpr = BLOCK_N * GROUP_N
    offs_ng = tl.arange(0, BLOCK_NG)

    # Load per-query auxiliary data once outside K loop (fixed per query block)
    ref_lse = tl.load(RefLSE + offs_m, mask=offs_m < num_queries, other=0.0)
    aux = tl.load(Aux + offs_m, mask=offs_m < num_queries, other=0.0)
    w = tl.load(W + offs_m, mask=offs_m < num_queries, other=0.0)

    # K-direction main loop: iterate over all documents in chunks of BLOCK_NG
    for start_n in range(0, num_docs, BLOCK_NG):
        # tl.multiple_of: hint to compiler that start_n is a multiple of BLOCK_NG
        # Enables more efficient address computation optimizations
        start_n = tl.multiple_of(start_n, BLOCK_NG)
        cur_offs_n = start_n + offs_ng

        # [Load labels tile] (same pattern as forward kernel)
        if INT64_LABELS:
            label_ptrs = (
                Labels + offs_m[:, None].to(tl.int64) * stride_l_m + cur_offs_n[None, :].to(tl.int64) * stride_l_n
            )
        else:
            label_ptrs = Labels + offs_m[:, None] * stride_l_m + cur_offs_n[None, :] * stride_l_n
        labels_tile = tl.load(
            label_ptrs, mask=(offs_m[:, None] < num_queries) & (cur_offs_n[None, :] < num_docs), other=-1
        )

        # [Recompute Q@K^T score tile] Same code as forward kernel to recompute scores
        # Core of fused approach: no stored score matrix, so must recompute in backward
        qk = tl.zeros([BLOCK_M, BLOCK_NG], dtype=tl.float32)
        for start_d in range(0, hidden_dim, BLOCK_D):
            cur_offs_d = start_d + offs_d
            q_ptrs = Q + offs_m[:, None] * stride_q_m + cur_offs_d[None, :] * stride_q_d
            k_ptrs = K + cur_offs_n[:, None] * stride_k_n + cur_offs_d[None, :] * stride_k_d
            q_tile = tl.load(
                q_ptrs, mask=(offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            k_tile = tl.load(
                k_ptrs, mask=(cur_offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            if FP32_MODE and not ALLOW_TF32:
                qk += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
            else:
                qk += tl.dot(q_tile, tl.trans(k_tile))

        # [Compute grad_s: loss gradient w.r.t. score]
        # Formula differs per loss type; tl.constexpr branching leaves only one path at compile time (zero runtime cost)
        #
        # scores_minus = score[i,j] - logsumexp[i]
        # Negative means exp(scores_minus) < 1, positive means > 1
        # softmax(score) = exp(scores_minus) computes softmax stably
        scores_minus = qk - ref_lse[:, None]

        if LOSS_TYPE == 0:
            # [MP-NCE loss gradient]
            # positive pair (label > 0):
            #   grad = w * (sigmoid(score - neg_lse) - 1)
            #   sigmoid near 1 -> grad ~ 0 (well-classified positive)
            #   sigmoid near 0 -> grad ~ -w (misclassified positive, large gradient)
            pos_mask = labels_tile > 0
            # negative pair (label == 0):
            #   grad = w * exp(score - neg_lse) * sum_weights
            #   exp(score - neg_lse) = softmax-like weight
            #   sum_weights = sum of sigmoid(pos_score - neg_lse) for all pos pairs
            neg_mask = labels_tile == 0
            # Non-neg positions set to -inf so exp=0
            exp_neg = tl.exp(tl.where(neg_mask, scores_minus, float("-inf")))
            neg_grad = w[:, None] * exp_neg * aux[:, None]
            # Non-pos positions set to 0 so sigmoid=0.5 (doesn't matter, masked out anyway)
            sig_pos = tl.sigmoid(tl.where(pos_mask, scores_minus, 0.0))
            pos_grad = w[:, None] * (sig_pos - 1.0)
            # pos -> pos_grad, neg -> neg_grad, ignore -> 0
            grad_s = tl.where(pos_mask, pos_grad, tl.where(neg_mask, neg_grad, 0.0))
        elif LOSS_TYPE == 1:
            # [Soft CE loss gradient]
            # valid pair (label >= 0):
            #   grad = w * (softmax(score) - normalized_label)
            #   softmax(score) = exp(score - all_lse)
            #   normalized_label = label / sum(labels)
            valid_mask = labels_tile >= 0
            softmax_val = tl.exp(tl.where(valid_mask, scores_minus, float("-inf")))
            # Convert label to float. Clamp -1 (ignore) to 0
            label_float = tl.maximum(labels_tile.to(tl.float32), 0.0)
            # Replace label_sum=0 with 1 to prevent division by zero
            safe_aux = tl.where(aux > 0, aux, 1.0)
            norm_label = label_float / safe_aux[:, None]
            grad_s = w[:, None] * (softmax_val - norm_label)
            # Invalid positions get zero gradient
            grad_s = tl.where(valid_mask, grad_s, 0.0)
        else:
            # [Cross CE loss gradient]
            # valid pair (label >= 0):
            #   grad = w * (label_sum * softmax(score) - label)
            valid_mask = labels_tile >= 0
            softmax_val = tl.exp(tl.where(valid_mask, scores_minus, float("-inf")))
            label_float = tl.maximum(labels_tile.to(tl.float32), 0.0)
            grad_s = w[:, None] * (aux[:, None] * softmax_val - label_float)
            grad_s = tl.where(valid_mask, grad_s, 0.0)

        # Zero out gradient for padding region (indices beyond actual query/doc count)
        grad_s = tl.where(cur_offs_n[None, :] < num_docs, grad_s, 0.0)
        grad_s = tl.where(offs_m[:, None] < num_queries, grad_s, 0.0)

        # [Accumulate dQ: dQ += grad_s @ K]
        # grad_s: [BLOCK_M, BLOCK_NG] score gradient
        # K_tile: [BLOCK_NG, BLOCK_D] document embedding slice
        # Result: [BLOCK_M, BLOCK_D] contribution to dQ
        #
        # If hidden_dim > BLOCK_D, also split along D direction
        for start_d in range(0, hidden_dim, BLOCK_D):
            cur_offs_d = start_d + offs_d
            # Load current doc block x dim block slice from K
            k_ptrs = K + cur_offs_n[:, None] * stride_k_n + cur_offs_d[None, :] * stride_k_d
            k_tile = tl.load(
                k_ptrs, mask=(cur_offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )

            # [Dtype casting for tensor cores]
            # tl.dot does not support direct fp32 x fp32 (not a tensor core format)
            # so inputs must be converted to a format tensor cores can process:
            #   CAST_DTYPE=0 (fp32 input): inline PTX asm to round fp32 -> tf32
            #     tf32 = fp32 with mantissa truncated to 10 bits for tensor core compatibility
            #   CAST_DTYPE=1 (bf16 input): cast grad_s (fp32) to bf16. k_tile is already bf16
            #   CAST_DTYPE=2 (fp16 input): cast grad_s (fp32) to fp16
            if CAST_DTYPE == 0:
                gs = tl.inline_asm_elementwise(ASM, "=r, r", [grad_s], dtype=tl.float32, is_pure=True, pack=1)
                kc = tl.inline_asm_elementwise(ASM, "=r, r", [k_tile], dtype=tl.float32, is_pure=True, pack=1)
            elif CAST_DTYPE == 1:
                gs = grad_s.to(tl.bfloat16)
                kc = k_tile
            else:
                gs = grad_s.to(tl.float16)
                kc = k_tile

            # [BLOCK_M, BLOCK_NG] @ [BLOCK_NG, BLOCK_D] -> [BLOCK_M, BLOCK_D]
            dq_contrib = tl.dot(gs, kc)
            # Load-add-store pattern:
            # Each K loop iteration adds a contribution to the same dQ positions,
            # so we load current dQ value, add the contribution, and store back.
            # Not atomic -- only this CTA writes to these dQ positions, no race condition
            dq_ptrs = dQ + offs_m[:, None] * stride_q_m + cur_offs_d[None, :] * stride_q_d
            dq_prev = tl.load(
                dq_ptrs, mask=(offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            tl.store(
                dq_ptrs, dq_prev + dq_contrib, mask=(offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim)
            )


# =============================================================================
# Backward: dK kernel (LOSS_TYPE branching)
# =============================================================================


@triton.autotune(configs=_bwd_dk_configs(), key=["hidden_dim"], reset_to_zero=["dK"])
@triton.jit
def _dk_bwd_kernel(
    # Tensor pointers
    Q,  # [num_queries, hidden_dim] query embeddings
    K,  # [num_docs, hidden_dim] document embeddings
    Labels,  # [num_queries, num_docs] int8 label matrix
    dK,  # [num_docs, hidden_dim] output: document gradient (accumulated in fp32)
    RefLSE,  # [num_queries] logsumexp from forward (same as dQ kernel)
    Aux,  # [num_queries] per-loss auxiliary value (same as dQ kernel)
    W,  # [num_queries] per-query loss weight
    # Dimension info
    num_queries,
    num_docs,
    hidden_dim,
    # stride
    stride_q_m,
    stride_q_d,  # Q (row, col) stride
    stride_k_n,
    stride_k_d,  # K, dK (row, col) stride
    stride_l_m,
    stride_l_n,  # Labels (row, col) stride
    # tl.constexpr
    BLOCK_M: tl.constexpr,  # Q-axis tile size. Number of queries seen per inner loop iteration
    BLOCK_N: tl.constexpr,  # K-axis tile size. Number of documents processed by one CTA
    BLOCK_D: tl.constexpr,  # hidden_dim-axis tile size
    GROUP_M: tl.constexpr,  # Q-axis tile group count (symmetric to GROUP_N in dQ)
    # GROUP_M=2 processes BLOCK_M*2 queries per loop iteration
    FP32_MODE: tl.constexpr,  # True if input is fp32
    ALLOW_TF32: tl.constexpr,  # True to allow tf32 tensorcore
    CAST_DTYPE: tl.constexpr,  # 0=tf32(asm), 1=bf16, 2=fp16
    LOSS_TYPE: tl.constexpr,  # 0=multi, 1=soft, 2=cross
    INT64_LABELS: tl.constexpr,  # True for int64 label pointer (Q*K > 2^31)
):
    """
    dK backward kernel: compute gradient for each document.

    Math:
      dK[j] = sum_i grad_s[i,j] * Q[i]
      i.e., grad_s^T [K,Q] @ Q [Q,D] -> dK [K,D]

    Difference from dQ kernel:
      dQ: CTAs partitioned along Q axis (each CTA handles BLOCK_M queries, iterates over K)
      dK: CTAs partitioned along K axis (each CTA handles BLOCK_N docs, iterates over Q)
      The rest (score recomputation, grad_s computation, accumulation) is identical.

    Parallelization:
      grid = (ceil(num_docs / BLOCK_N),)
      Each CTA computes dK for BLOCK_N documents. Q direction is sequential loop within CTA.
    """
    # PTX inline assembly for TF32 conversion (same as dQ kernel)
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"

    # Document indices assigned to this CTA
    # In dQ, pid handled queries; in dK, it handles documents
    pid_n = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offs_d = tl.arange(0, BLOCK_D)

    # GROUP_M: Q-axis tile group. Symmetric to GROUP_N in dQ.
    # Groups multiple Q tiles to reduce K tile reload count
    BLOCK_MG: tl.constexpr = BLOCK_M * GROUP_M
    offs_mg = tl.arange(0, BLOCK_MG)

    # Q-direction main loop: iterate over all queries in chunks of BLOCK_MG
    # In dQ this was a K-direction loop, in dK it's a Q-direction loop
    for start_m in range(0, num_queries, BLOCK_MG):
        # Hint to compiler that start_m is a multiple of BLOCK_MG for address optimization
        start_m = tl.multiple_of(start_m, BLOCK_MG)
        cur_offs_m = start_m + offs_mg

        # Load per-query auxiliary data.
        # In dQ kernel these were loaded once outside the loop, but in dK kernel
        # we iterate over Q, so must reload ref_lse, aux, w for each new query tile
        ref_lse = tl.load(RefLSE + cur_offs_m, mask=cur_offs_m < num_queries, other=0.0)
        aux = tl.load(Aux + cur_offs_m, mask=cur_offs_m < num_queries, other=0.0)
        w = tl.load(W + cur_offs_m, mask=cur_offs_m < num_queries, other=0.0)

        # [Load labels tile] (same as dQ kernel, indices are cur_offs_m/offs_n)
        if INT64_LABELS:
            label_ptrs = (
                Labels + cur_offs_m[:, None].to(tl.int64) * stride_l_m + offs_n[None, :].to(tl.int64) * stride_l_n
            )
        else:
            label_ptrs = Labels + cur_offs_m[:, None] * stride_l_m + offs_n[None, :] * stride_l_n
        labels_tile = tl.load(
            label_ptrs, mask=(cur_offs_m[:, None] < num_queries) & (offs_n[None, :] < num_docs), other=-1
        )

        # [Recompute Q@K^T score tile] Same as forward/dQ
        qk = tl.zeros([BLOCK_MG, BLOCK_N], dtype=tl.float32)
        for start_d in range(0, hidden_dim, BLOCK_D):
            cur_offs_d = start_d + offs_d
            q_ptrs = Q + cur_offs_m[:, None] * stride_q_m + cur_offs_d[None, :] * stride_q_d
            k_ptrs = K + offs_n[:, None] * stride_k_n + cur_offs_d[None, :] * stride_k_d
            q_tile = tl.load(
                q_ptrs, mask=(cur_offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            k_tile = tl.load(k_ptrs, mask=(offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim), other=0.0)
            if FP32_MODE and not ALLOW_TF32:
                qk += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
            else:
                qk += tl.dot(q_tile, tl.trans(k_tile))

        # [Compute grad_s] Identical formulas to dQ kernel.
        # Same gradient for the same score, but accumulation direction differs (dQ multiplies by K, dK by Q)
        scores_minus = qk - ref_lse[:, None]

        if LOSS_TYPE == 0:
            # MP-NCE (see dQ kernel comments for details)
            pos_mask = labels_tile > 0
            neg_mask = labels_tile == 0
            exp_neg = tl.exp(tl.where(neg_mask, scores_minus, float("-inf")))
            neg_grad = w[:, None] * exp_neg * aux[:, None]
            sig_pos = tl.sigmoid(tl.where(pos_mask, scores_minus, 0.0))
            pos_grad = w[:, None] * (sig_pos - 1.0)
            grad_s = tl.where(pos_mask, pos_grad, tl.where(neg_mask, neg_grad, 0.0))
        elif LOSS_TYPE == 1:
            # Soft CE (see dQ kernel comments for details)
            valid_mask = labels_tile >= 0
            softmax_val = tl.exp(tl.where(valid_mask, scores_minus, float("-inf")))
            label_float = tl.maximum(labels_tile.to(tl.float32), 0.0)
            safe_aux = tl.where(aux > 0, aux, 1.0)
            norm_label = label_float / safe_aux[:, None]
            grad_s = w[:, None] * (softmax_val - norm_label)
            grad_s = tl.where(valid_mask, grad_s, 0.0)
        else:
            # Cross CE (see dQ kernel comments for details)
            valid_mask = labels_tile >= 0
            softmax_val = tl.exp(tl.where(valid_mask, scores_minus, float("-inf")))
            label_float = tl.maximum(labels_tile.to(tl.float32), 0.0)
            grad_s = w[:, None] * (aux[:, None] * softmax_val - label_float)
            grad_s = tl.where(valid_mask, grad_s, 0.0)

        # Zero out gradient for padding region
        grad_s = tl.where(cur_offs_m[:, None] < num_queries, grad_s, 0.0)
        grad_s = tl.where(offs_n[None, :] < num_docs, grad_s, 0.0)

        # [Accumulate dK: dK += grad_s^T @ Q]
        # In dQ it was grad_s @ K, but for dK we transpose grad_s and multiply by Q
        # grad_s shape: [BLOCK_MG, BLOCK_N]
        # grad_s^T shape: [BLOCK_N, BLOCK_MG]
        # Q_tile shape: [BLOCK_MG, BLOCK_D]
        # Result: [BLOCK_N, BLOCK_D] = contribution to this doc block's dK
        for start_d in range(0, hidden_dim, BLOCK_D):
            cur_offs_d = start_d + offs_d
            # Load current query block x dim block slice from Q
            q_ptrs = Q + cur_offs_m[:, None] * stride_q_m + cur_offs_d[None, :] * stride_q_d
            q_tile = tl.load(
                q_ptrs, mask=(cur_offs_m[:, None] < num_queries) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )

            # Tensor core dtype casting (same pattern as dQ kernel)
            if CAST_DTYPE == 0:
                gs = tl.inline_asm_elementwise(ASM, "=r, r", [grad_s], dtype=tl.float32, is_pure=True, pack=1)
                qc = tl.inline_asm_elementwise(ASM, "=r, r", [q_tile], dtype=tl.float32, is_pure=True, pack=1)
            elif CAST_DTYPE == 1:
                gs = grad_s.to(tl.bfloat16)
                qc = q_tile
            else:
                gs = grad_s.to(tl.float16)
                qc = q_tile

            # tl.trans(gs): transpose grad_s from [BLOCK_MG, BLOCK_N] to [BLOCK_N, BLOCK_MG]
            # [BLOCK_N, BLOCK_MG] @ [BLOCK_MG, BLOCK_D] -> [BLOCK_N, BLOCK_D]
            dk_contrib = tl.dot(tl.trans(gs), qc)
            # Load-add-store pattern to accumulate into dK.
            # Only this CTA writes to this doc block, so no race condition
            dk_ptrs = dK + offs_n[:, None] * stride_k_n + cur_offs_d[None, :] * stride_k_d
            dk_prev = tl.load(
                dk_ptrs, mask=(offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim), other=0.0
            )
            tl.store(
                dk_ptrs, dk_prev + dk_contrib, mask=(offs_n[:, None] < num_docs) & (cur_offs_d[None, :] < hidden_dim)
            )


# =============================================================================
# Python Helpers
# =============================================================================


def _get_dtype_params(dtype):
    """
    Determine Triton kernel parameters based on input tensor dtype.

    Returns:
        fp32_mode: True if input is fp32. Used for tl.dot precision branching in kernel
        cast_dtype: casting method before passing to tl.dot in backward kernels
                    0 = tf32 (inline asm to round fp32 to tf32)
                    1 = bf16 (cast grad_s to bf16)
                    2 = fp16 (cast grad_s to fp16)
    """
    if dtype == torch.float32:
        return True, 0  # FP32_MODE=True, CAST_DTYPE=tf32
    elif dtype == torch.bfloat16:
        return False, 1  # FP32_MODE=False, CAST_DTYPE=bf16
    else:
        return False, 2  # FP32_MODE=False, CAST_DTYPE=fp16


def _select_group_n(num_docs):
    """Determine K-axis tile group count. Group by 2 when enough docs to reduce Q reloads."""
    return 2 if num_docs >= _GN_THRESHOLD else 1


def _select_group_m(num_queries):
    """Determine Q-axis tile group count. Group by 2 when enough queries to reduce K reloads."""
    return 2 if num_queries >= _GM_THRESHOLD else 1


# =============================================================================
# Python Wrappers for Triton Kernels
# =============================================================================


def _lse_forward(q_scaled, k, labels, lse_mode, allow_tf32=False):
    """
    Python wrapper for the forward LSE kernel.

    Makes tensors contiguous, allocates output tensor, computes kernel parameters,
    and launches the Triton kernel.

    Args:
        q_scaled: [Q, D] scale-multiplied query embeddings
        k: [K, D] document embeddings
        labels: [Q, K] int8 label matrix
        lse_mode: LSE_NEG_ONLY(0) or LSE_VALID_ALL(1)
        allow_tf32: whether to allow tf32 tensorcore for fp32 input

    Returns:
        [Q] logsumexp value for each query (fp32)
    """
    num_queries, hidden_dim = q_scaled.shape
    num_docs = k.shape[0]

    # Round up output tensor size to multiple of MAX_BLOCK_M(=64).
    # The kernel processes in BLOCK_M chunks; excess queries are masked inside the kernel
    MAX_BLOCK_M = 64
    num_queries_rounded = math.ceil(num_queries / MAX_BLOCK_M) * MAX_BLOCK_M
    out_lse = torch.empty(num_queries_rounded, device=q_scaled.device, dtype=torch.float32)

    # Triton kernels assume contiguous memory layout for stride calculations.
    # Non-contiguous tensors would cause incorrect stride values and wrong memory reads
    q_cont = q_scaled.contiguous()
    k_cont = k.contiguous()
    labels_cont = labels.contiguous()
    fp32_mode, _ = _get_dtype_params(q_cont.dtype)

    # Q size bucket (for autotune cache separation)
    q_bucket = _bucket_q(num_queries)
    # Labels row stride = num_docs; query_idx * num_docs can overflow int32 max.
    # Pass this flag to kernel to use int64 pointer arithmetic
    int64_labels = (num_queries * num_docs) > (2**31 - 1)

    # grid: number of CTAs (thread blocks) to launch. triton.cdiv = ceiling division
    # autotune determines BLOCK_M, so use lambda for deferred computation
    grid = lambda META: (triton.cdiv(num_queries, META["BLOCK_M"]),)
    _lse_fwd_kernel[grid](
        q_cont,
        k_cont,
        labels_cont,
        out_lse,
        num_queries,
        num_docs,
        hidden_dim,
        q_bucket,
        # .stride(0)/.stride(1): automatically pass the PyTorch tensor's row/col stride
        q_cont.stride(0),
        q_cont.stride(1),
        k_cont.stride(0),
        k_cont.stride(1),
        labels_cont.stride(0),
        labels_cont.stride(1),
        GROUP_N=_select_group_n(num_docs),
        FP32_MODE=fp32_mode,
        ALLOW_TF32=allow_tf32,
        LSE_MODE=lse_mode,
        INT64_LABELS=int64_labels,
    )
    # Trim the rounded-up portion, returning only actual query count
    return out_lse[:num_queries]


def _neg_lse_forward(q_scaled, k, labels, allow_tf32=False):
    """MP-NCE wrapper: compute logsumexp using only negative pair (labels==0) scores."""
    return _lse_forward(q_scaled, k, labels, LSE_NEG_ONLY, allow_tf32=allow_tf32)


def _all_lse_forward(q_scaled, k, labels, allow_tf32=False):
    """CE wrapper: compute logsumexp using all valid pair (labels>=0, pos+neg) scores."""
    return _lse_forward(q_scaled, k, labels, LSE_VALID_ALL, allow_tf32=allow_tf32)


def _backward(q_scaled, k, labels, ref_lse, aux, w, loss_type_int, allow_tf32=False):
    """
    Backward wrapper: sequentially launch dQ and dK kernels to compute gradients.

    The two kernels are independent and could in principle run concurrently,
    but currently launched sequentially (auto-serialized on a single CUDA stream).

    Args:
        q_scaled: [Q, D] scale-multiplied query
        k: [K, D] document
        labels: [Q, K] int8 label
        ref_lse: [Q] logsumexp computed in forward
        aux: [Q] per-loss auxiliary value (sum_weights or label_sum)
        w: [Q] per-query weight (grad_output * inv_weight)
        loss_type_int: 0=multi, 1=soft, 2=cross
        allow_tf32: allow tf32 tensorcore

    Returns:
        dq: [Q, D] query gradient (fp32)
        dk: [K, D] document gradient (fp32)
    """
    num_queries, hidden_dim = q_scaled.shape
    num_docs = k.shape[0]

    q_cont = q_scaled.contiguous()
    k_cont = k.contiguous()
    labels_cont = labels.contiguous()

    # dq, dk allocated as empty — Triton autotune's reset_to_zero=["dQ"]/["dK"] on
    # _dq_bwd_kernel/_dk_bwd_kernel zeros these in the kernel prologue, so an
    # explicit zero-init here would be a redundant aten::zero_ launch.
    dq = torch.empty(q_cont.shape, device=q_cont.device, dtype=torch.float32)
    dk = torch.empty(k_cont.shape, device=k_cont.device, dtype=torch.float32)

    ref_lse_c = ref_lse.contiguous()
    aux_c = aux.contiguous()
    w_c = w.contiguous()

    fp32_mode, cast_dtype = _get_dtype_params(q_cont.dtype)

    q_bucket = _bucket_q(num_queries)
    # int64 label pointer flag (same condition as _lse_forward)
    int64_labels = (num_queries * num_docs) > (2**31 - 1)

    # dQ kernel: grid = ceil(Q / BLOCK_M). Each CTA computes gradient for BLOCK_M queries
    grid_q = lambda META: (triton.cdiv(num_queries, META["BLOCK_M"]),)
    _dq_bwd_kernel[grid_q](
        q_cont,
        k_cont,
        labels_cont,
        dq,
        ref_lse_c,
        aux_c,
        w_c,
        num_queries,
        num_docs,
        hidden_dim,
        q_bucket,
        q_cont.stride(0),
        q_cont.stride(1),
        k_cont.stride(0),
        k_cont.stride(1),
        labels_cont.stride(0),
        labels_cont.stride(1),
        GROUP_N=_select_group_n(num_docs),
        FP32_MODE=fp32_mode,
        ALLOW_TF32=allow_tf32,
        CAST_DTYPE=cast_dtype,
        LOSS_TYPE=loss_type_int,
        INT64_LABELS=int64_labels,
    )

    # dK kernel: grid = ceil(K / BLOCK_N). Each CTA computes gradient for BLOCK_N documents
    # dK kernel doesn't need q_bucket (parallelized along K axis, Q size doesn't affect autotune key)
    grid_k = lambda META: (triton.cdiv(num_docs, META["BLOCK_N"]),)
    _dk_bwd_kernel[grid_k](
        q_cont,
        k_cont,
        labels_cont,
        dk,
        ref_lse_c,
        aux_c,
        w_c,
        num_queries,
        num_docs,
        hidden_dim,
        q_cont.stride(0),
        q_cont.stride(1),
        k_cont.stride(0),
        k_cont.stride(1),
        labels_cont.stride(0),
        labels_cont.stride(1),
        GROUP_M=_select_group_m(num_queries),
        FP32_MODE=fp32_mode,
        ALLOW_TF32=allow_tf32,
        CAST_DTYPE=cast_dtype,
        LOSS_TYPE=loss_type_int,
        INT64_LABELS=int64_labels,
    )

    return dq[:num_queries], dk[:num_docs]


# =============================================================================
# Positive Pair Helpers
# =============================================================================


def _resolve_positive_pairs(q_scaled, k, labels, pos_qi, pos_di, pos_counts, neg_counts):
    """
    Prepare positive pair indices and related info.

    The caller (loss.py) may pass pre-computed pos_qi/pos_di, or if not provided,
    they are extracted directly from the labels matrix.

    Args:
        q_scaled: [Q, D] query embeddings (for score computation)
        k: [K, D] document embeddings
        labels: [Q, K] int8 label matrix
        pos_qi: [P] positive pair query indices (None to extract from labels)
        pos_di: [P] positive pair document indices
        pos_counts: [Q] per-query positive count (None to compute from pos_qi via bincount)
        neg_counts: [Q] per-query negative count (None to estimate)

    Returns:
        pos_qi: [P] positive pair query indices
        pos_di: [P] positive pair document indices
        num_pos: [Q] per-query positive count
        has_neg: [Q] bool, whether each query has negatives
        pos_scores: [P] dot product score for each positive pair (no grad)
        pos_label_values: [P] label value for each positive pair (float32)
    """
    num_queries = q_scaled.shape[0]
    num_docs = k.shape[0]
    device = q_scaled.device

    if pos_qi is not None and pos_di is not None:
        # Use pre-computed positive pair indices from caller
        pos_qi = pos_qi.to(device, non_blocking=True)
        pos_di = pos_di.to(device, non_blocking=True)
        num_pos = (
            pos_counts.to(device, non_blocking=True).to(torch.int64)
            if pos_counts is not None
            # If pos_counts not provided, count occurrences of each query in pos_qi
            else torch.bincount(pos_qi, minlength=num_queries).to(torch.int64)
        )
    else:
        # Extract positive pairs directly from labels matrix (positions where label > 0)
        pos_mask = labels > 0
        num_pos = pos_mask.sum(dim=1)
        # torch.where: returns (row, col) index pairs where condition is True
        pos_qi, pos_di = torch.where(pos_mask)

    # Negative existence: in MP-NCE, queries without negatives are excluded from loss
    if neg_counts is not None:
        has_neg = neg_counts.to(device, non_blocking=True).to(torch.int64) > 0
    else:
        # Assumes in_batch_negative=True: if fewer positives than total docs, rest are negatives
        has_neg = num_pos < num_docs

    # Compute positive scores (in Python, no_grad)
    # The Triton kernel recomputes the full score matrix, but positive pair scores
    # are few (P << Q*K) so computing them in Python is more efficient.
    # no_grad + detach: computed outside FusedDenseLoss.apply(), so must not
    # connect to autograd graph. Backward is handled directly by the Triton kernels
    with torch.no_grad():
        # q_scaled[pos_qi] * k[pos_di]: [P, D] element-wise product
        # .sum(dim=1): [P] dot product scores
        pos_scores = (q_scaled[pos_qi] * k[pos_di]).sum(dim=1, dtype=torch.float32)
        pos_label_values = labels[pos_qi, pos_di].to(torch.float32)

    return pos_qi, pos_di, num_pos, has_neg, pos_scores, pos_label_values


# =============================================================================
# Autograd Function
# =============================================================================


class FusedDenseLoss(torch.autograd.Function):
    """
    Fused Dense Loss implemented as a PyTorch autograd Function.

    Inheriting from torch.autograd.Function allows defining custom forward/backward.
    Normally PyTorch auto-generates backward, but this kernel doesn't store the score
    matrix, so backward must recompute Q@K^T. Hence both forward and backward are
    manually implemented.

    ctx: context object for passing tensors from forward to backward.
         Use save_for_backward() to save, and saved_tensors to retrieve in backward.
    """

    @staticmethod
    def forward(
        ctx,
        q_scaled: Tensor,  # [Q, D] scale-multiplied query
        k: Tensor,  # [K, D] document
        labels: Tensor,  # [Q, K] int8 label
        loss_type_int: int,  # 0=multi, 1=soft, 2=cross
        allow_tf32: bool,  # whether to allow tf32
        pos_qi: Tensor,  # [P] positive pair query indices
        pos_di: Tensor,  # [P] positive pair document indices
        num_pos: Tensor,  # [Q] per-query positive count
        has_neg: Tensor,  # [Q] bool, negative existence
        pos_scores: Tensor,  # [P] positive pair scores (pre-computed in Python)
        pos_label_values: Tensor,  # [P] positive pair label values
    ):
        """
        Unified forward for all 3 loss types.

        Common flow:
          1) Compute logsumexp via Triton kernel (no score matrix stored)
          2) Combine positive pair scores with logsumexp in Python to compute loss
          3) Save tensors needed for backward (ref_lse, aux, w)

        MP-NCE: neg_lse kernel (negatives only) -> softplus loss
        Soft/Cross CE: all_lse kernel (pos+neg) -> CE loss
        """
        num_queries = q_scaled.shape[0]
        device = q_scaled.device

        # Per-loss-type forward computation + prepare auxiliary tensors for backward
        if loss_type_int == LOSS_MULTI:
            # === MP-NCE ===
            # 1) Compute logsumexp of negative scores (Triton kernel)
            neg_lse = _neg_lse_forward(q_scaled, k, labels, allow_tf32=allow_tf32)
            # Get neg_lse for each positive pair's query
            pos_neg_lse = neg_lse[pos_qi]

            # Valid query: only compute loss for queries that have both positive and negative pairs
            valid = (num_pos > 0) & has_neg
            valid_f = valid.to(torch.float32)
            # Loss average denominator: valid query count (clamp to 1 to prevent div by zero)
            denom = valid_f.sum().clamp(min=1.0)
            # Per-query positive count (clamp to 1 for averaging, prevent div by zero)
            num_pos_f = num_pos.clamp(min=1).to(torch.float32)

            # MP-NCE loss = softplus(neg_lse - pos_score)
            # softplus(x) = log(1 + exp(x)). Smooth ReLU.
            # If neg_lse > pos_score, loss is large; if neg_lse < pos_score, loss is small
            per_pos_loss = F.softplus(pos_neg_lse - pos_scores)
            # Sum losses per query (a query may have multiple positive pairs)
            query_loss_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            query_loss_sum.scatter_add_(0, pos_qi, per_pos_loss)
            # Per-query average -> select valid only -> overall average
            loss = ((query_loss_sum / num_pos_f) * valid_f).sum() / denom

            # Prepare auxiliary tensors for backward
            # aux = sum_weights: per-query sum of sigma(pos_score - neg_lse)
            # sigmoid values are used in dQ/dK kernels for grad_s computation
            sigmoid_vals = torch.sigmoid(pos_neg_lse - pos_scores)
            aux = torch.zeros(num_queries, device=device, dtype=torch.float32)
            aux.scatter_add_(0, pos_qi, sigmoid_vals)
            ref_lse = neg_lse
            # inv_weight: per-query weight to multiply grad_s in backward
            inv_weight = valid_f / (denom * num_pos_f)

        elif loss_type_int == LOSS_SOFT:
            # === Soft Label CE ===
            # 1) Compute logsumexp of all valid scores (pos + neg, excluding ignore)
            all_lse = _all_lse_forward(q_scaled, k, labels, allow_tf32=allow_tf32)

            # Per-query label sum (soft labels, so sum of label values where label > 0)
            label_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            label_sum.scatter_add_(0, pos_qi, pos_label_values)
            valid = label_sum > 0
            valid_f = valid.to(torch.float32)
            denom = valid_f.sum().clamp(min=1.0)

            # Weighted average of positive scores using normalized labels
            # norm_label = label / label_sum (normalized to probability distribution)
            safe_label_sum = label_sum[pos_qi].clamp(min=1e-9)
            norm_labels = pos_label_values / safe_label_sum
            weighted_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            weighted_sum.scatter_add_(0, pos_qi, norm_labels * pos_scores)

            # Soft CE loss = logsumexp - weighted_positive_score
            query_losses = all_lse - weighted_sum
            # NaN guard: for invalid queries, all_lse can be -inf
            # -inf - 0 = -inf is fine, but 0 * -inf = NaN, so use where to handle
            query_losses = torch.where(valid, query_losses, torch.zeros_like(query_losses))
            loss = query_losses.sum() / denom

            # Backward auxiliary tensors
            ref_lse = all_lse
            aux = label_sum
            inv_weight = valid_f / denom

        else:
            # === Cross CE ===
            # Similar to Soft CE but uses raw label values without normalization
            all_lse = _all_lse_forward(q_scaled, k, labels, allow_tf32=allow_tf32)

            label_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            label_sum.scatter_add_(0, pos_qi, pos_label_values)
            valid = label_sum > 0
            valid_f = valid.to(torch.float32)
            denom = valid_f.sum().clamp(min=1.0)

            # Label-weighted positive scores (no normalization)
            weighted_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            weighted_sum.scatter_add_(0, pos_qi, pos_label_values * pos_scores)

            # Cross CE loss = label_sum * logsumexp - weighted_positive_score
            query_losses = label_sum * all_lse - weighted_sum
            # NaN guard: prevent 0 * (-inf) = NaN when label_sum=0
            query_losses = torch.where(valid, query_losses, torch.zeros_like(query_losses))
            loss = query_losses.sum() / denom

            # Backward auxiliary tensors
            ref_lse = all_lse
            aux = label_sum
            inv_weight = valid_f / denom

        # Save tensors for backward
        # PyTorch keeps these in memory until backward completes
        ctx.save_for_backward(q_scaled, k, labels, ref_lse, aux, inv_weight)
        ctx.allow_tf32 = allow_tf32
        ctx.loss_type_int = loss_type_int
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: launch Triton dQ/dK kernels to compute gradients.

        grad_output: upstream gradient w.r.t. loss (scalar, usually 1.0)
        w = grad_output * inv_weight: final per-query weight passed to kernels
        """
        q_scaled, k, labels, ref_lse, aux, inv_weight = ctx.saved_tensors
        # Multiply grad_output (scalar) with per-query inv_weight to get per-query weight
        # This w is multiplied into grad_s inside the kernels
        w = grad_output * inv_weight

        # Launch Triton dQ/dK kernels (computed in fp32)
        dq_scaled, dk = _backward(
            q_scaled,
            k,
            labels,
            ref_lse,
            aux,
            w,
            loss_type_int=ctx.loss_type_int,
            allow_tf32=ctx.allow_tf32,
        )

        # Cast fp32 gradients back to original input dtype
        dq_scaled = dq_scaled.to(q_scaled.dtype)
        dk = dk.to(k.dtype)
        # Must return gradients for all forward arguments (11 total)
        # Only q_scaled and k have gradients; the rest are None
        return dq_scaled, dk, None, None, None, None, None, None, None, None, None


# =============================================================================
# Public API
# =============================================================================


def fused_dense_loss(
    q: Tensor,
    k: Tensor,
    labels: Tensor,
    scale: "float | Tensor",
    loss_type: str = "multi",
    allow_tf32: bool = False,
    pos_qi: Optional[Tensor] = None,
    pos_di: Optional[Tensor] = None,
    pos_counts: Optional[Tensor] = None,
    neg_counts: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Dense Embedding Loss (all 3 types unified) - main entry point.

    Standard contrastive loss first computes score_matrix = Q @ K^T and stores it,
    then computes loss. This function avoids storing the score matrix by recomputing
    it tile-by-tile inside Triton kernels, saving memory.

    Recommended to check should_use_fused() beforehand to verify fused is beneficial.
    For small Q*K, the reference (score matrix approach) may be faster.

    Args:
        q: [Q, D] query embeddings (normalized)
        k: [K, D] document embeddings (normalized)
        labels: [Q, K] int8 labels (>0: positive, 0: negative, -1: ignore)
        scale: temperature scale to multiply scores (float or learnable Tensor)
               q_scaled = q * scale before passing to kernel
        loss_type: "multi" (MP-NCE), "soft" (Soft CE), "cross" (Cross CE)
        allow_tf32: whether to allow TF32 tensorcore in fp32 mode (precision vs speed tradeoff)
        pos_qi: [P] int64, positive pair query indices (optional, extracted from labels if None)
        pos_di: [P] int64, positive pair doc indices (optional)
        pos_counts: [Q] int64, per-query positive count (optional)
        neg_counts: [Q] int64, per-query negative count (optional)

    Returns:
        scalar loss tensor (gradient-tracked, .backward() callable)
    """
    # Convert loss_type string to integer (for tl.constexpr branching in kernels)
    loss_type_int = _LOSS_TYPE_MAP.get(loss_type, LOSS_MULTI)

    # Apply temperature scale. If scale is a learnable Tensor, connects to autograd graph
    q_scaled = q * scale

    # Prepare positive pair info (indices, scores, label values)
    pos_qi, pos_di, num_pos, has_neg, pos_scores, pos_label_values = _resolve_positive_pairs(
        q_scaled, k, labels, pos_qi, pos_di, pos_counts, neg_counts
    )

    # Launch via FusedDenseLoss.apply() to use the autograd Function
    # .apply() calls forward, and .backward() triggers backward
    return FusedDenseLoss.apply(
        q_scaled,
        k,
        labels,
        loss_type_int,
        allow_tf32,
        pos_qi,
        pos_di,
        num_pos,
        has_neg,
        pos_scores,
        pos_label_values,
    )
