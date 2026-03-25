"""InfoNCE (NT-Xent) contrastive loss."""

import torch
import torch.nn.functional as F


def infonce_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
                 temperature: float = 0.02) -> torch.Tensor:
    """Compute InfoNCE loss for contrastive learning.

    Positives are along the diagonal (query[i] matches doc[i]).

    Args:
        query_embeds: [B, D] normalized query embeddings.
        doc_embeds: [B, D] normalized document embeddings.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss tensor.
    """
    sim = torch.matmul(query_embeds, doc_embeds.t()) / temperature  # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)
