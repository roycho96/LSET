"""Pooling strategies for packed (variable-length) hidden states."""

import torch
import torch.nn.functional as F

from ..kernels.fused_normalize import normalize as _normalize


def packed_pool(hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                strategy: str, normalize: bool = True) -> torch.Tensor:
    """Pool packed hidden states into per-sequence embeddings.

    Args:
        hidden_states: (total_tokens, H)
        cu_seqlens: (num_seqs + 1,) int32
        strategy: "last_token" | "mean" | "cls"
        normalize: Whether to L2-normalize output.

    Returns:
        (num_seqs, H) embeddings
    """
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
            torch.arange(num_seqs, device=hidden_states.device), lengths,
        )
        emb = torch.zeros(num_seqs, H, dtype=hidden_states.dtype,
                          device=hidden_states.device)
        emb.scatter_add_(0, seq_ids.unsqueeze(-1).expand_as(hidden_states),
                         hidden_states)
        emb = emb / lengths.unsqueeze(-1).to(emb.dtype).clamp(min=1e-9)
    else:
        raise ValueError(f"Unknown packed pooling strategy: {strategy}")

    if normalize:
        emb = _normalize(emb)

    return emb
