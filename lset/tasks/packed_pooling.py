"""Pooling strategies for packed (variable-length) hidden states."""

import torch

from lset.kernels.normalize import normalize as _normalize
from lset.kernels.pool_normalize import fused_pool_normalize as _fused_pool_normalize
from lset.kernels.segment_pool import triton_segment_mean_pool as _triton_segment_mean


def packed_pool(
    hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, strategy: str, normalize: bool = True
) -> torch.Tensor:
    """Pool packed hidden states into per-sequence embeddings.

    Args:
        hidden_states: (total_tokens, H)
        cu_seqlens: (num_seqs + 1,) int32
        strategy: "last_token" | "mean" | "cls"
        normalize: Whether to L2-normalize output.

    Returns:
        (num_seqs, H) embeddings
    """
    # Fused path: pool + normalize in one shot when on CUDA
    import os

    if os.environ.get("LSET_DISABLE_FUSED_POOL_NORMALIZE") != "1" and hidden_states.is_cuda:
        if strategy == "mean":
            # Triton segment kernel: mean + optional normalize in one kernel
            return _triton_segment_mean(hidden_states, cu_seqlens, normalize)
        if normalize and strategy in ("last_token", "cls"):
            return _fused_pool_normalize(hidden_states, cu_seqlens, strategy)

    num_seqs = cu_seqlens.shape[0] - 1

    if strategy == "last_token":
        indices = (cu_seqlens[1:] - 1).long()
        emb = hidden_states[indices]
    elif strategy == "cls":
        indices = cu_seqlens[:-1].long()
        emb = hidden_states[indices]
    elif strategy == "mean":
        H = hidden_states.shape[-1]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device),
            lengths,
        )
        emb = torch.zeros(num_seqs, H, dtype=hidden_states.dtype, device=hidden_states.device)
        emb.scatter_add_(0, seq_ids.unsqueeze(-1).expand_as(hidden_states), hidden_states)
        emb = emb / lengths.unsqueeze(-1).to(emb.dtype).clamp(min=1e-9)
    else:
        raise ValueError(f"Unknown packed pooling strategy: {strategy}")

    if normalize:
        emb = _normalize(emb)

    return emb
