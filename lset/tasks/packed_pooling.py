"""Pooling strategies for packed (variable-length) hidden states."""

import torch
import torch.nn.functional as F


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
        emb = torch.zeros(num_seqs, H, dtype=hidden_states.dtype,
                          device=hidden_states.device)
        for i in range(num_seqs):
            s = int(cu_seqlens[i])
            e = int(cu_seqlens[i + 1])
            emb[i] = hidden_states[s:e].mean(0)
    else:
        raise ValueError(f"Unknown packed pooling strategy: {strategy}")

    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)

    return emb
