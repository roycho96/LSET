"""Pooling strategies for embedding extraction."""

import torch
import torch.nn.functional as F

from ..kernels.fused_normalize import normalize as _normalize


def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
         strategy: str, normalize: bool = True) -> torch.Tensor:
    """Pool hidden states into a single embedding vector per sequence.

    Args:
        hidden_states: [B, S, H] tensor of hidden states.
        attention_mask: [B, S] tensor (1 for real tokens, 0 for padding).
        strategy: One of "last_token", "mean", "cls", "weighted_mean".
        normalize: Whether to L2-normalize the output embeddings.

    Returns:
        [B, H] tensor of pooled embeddings.
    """
    if strategy == "last_token":
        emb = hidden_states[:, -1, :]
    elif strategy == "cls":
        emb = hidden_states[:, 0, :]
    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    elif strategy == "weighted_mean":
        mask = attention_mask.float()  # [B, S]
        weights = torch.cumsum(mask, dim=1) * mask  # positions 1,2,3,...
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        emb = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")

    if normalize:
        emb = _normalize(emb)

    return emb
