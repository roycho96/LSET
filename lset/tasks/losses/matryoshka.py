"""Matryoshka Representation Learning (MRL) loss."""

import torch
import torch.nn.functional as F

from .infonce import infonce_loss


def matryoshka_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
                    dims: list[int], temperature: float = 0.02) -> torch.Tensor:
    """Compute MRL loss: average InfoNCE over multiple embedding dimensions.

    Args:
        query_embeds: [B, D] query embeddings (NOT yet normalized).
        doc_embeds: [B, D] document embeddings (NOT yet normalized).
        dims: List of truncation dimensions, e.g. [64, 128, 256, 768].
        temperature: Temperature for InfoNCE.

    Returns:
        Scalar loss (mean over all dims).
    """
    total_loss = torch.tensor(0.0, device=query_embeds.device)
    for d in dims:
        q = F.normalize(query_embeds[:, :d], p=2, dim=-1)
        doc = F.normalize(doc_embeds[:, :d], p=2, dim=-1)
        total_loss = total_loss + infonce_loss(q, doc, temperature)
    return total_loss / len(dims)
