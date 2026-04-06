"""Matryoshka Representation Learning (MRL) loss."""

import torch
import torch.nn.functional as F

from lset.losses.infonce import infonce_loss
from lset.losses.contrastive import contrastive_loss


def matryoshka_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
                    dims: list[int], temperature: float = 0.02,
                    labels: torch.Tensor | None = None) -> torch.Tensor:
    """Compute MRL loss: average contrastive loss over multiple embedding dimensions.

    Args:
        query_embeds: [B, D] query embeddings (NOT yet normalized).
        doc_embeds: [B, D] document embeddings (NOT yet normalized).
        dims: List of truncation dimensions, e.g. [64, 128, 256, 768].
        temperature: Temperature for contrastive loss.
        labels: Optional (Q, K) label matrix. If None, uses diagonal InfoNCE.

    Returns:
        Scalar loss (mean over all dims).
    """
    total_loss = torch.tensor(0.0, device=query_embeds.device)
    for d in dims:
        q = F.normalize(query_embeds[:, :d], p=2, dim=-1)
        doc = F.normalize(doc_embeds[:, :d], p=2, dim=-1)
        if labels is not None:
            total_loss = total_loss + contrastive_loss(q, doc, labels, temperature)
        else:
            total_loss = total_loss + infonce_loss(q, doc, temperature)
    return total_loss / len(dims)
