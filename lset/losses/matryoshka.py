"""Matryoshka Representation Learning (MRL) loss."""

import torch
import torch.nn.functional as F

from lset.losses.contrastive import contrastive_loss
from lset.losses.infonce import infonce_loss


def matryoshka_loss(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    dims: list[int],
    temperature: float = 0.02,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MRL loss: average contrastive loss over multiple embedding dimensions."""
    total_loss = torch.tensor(0.0, device=query_embeds.device)
    for d in dims:
        q = F.normalize(query_embeds[:, :d], p=2, dim=-1)
        doc = F.normalize(doc_embeds[:, :d], p=2, dim=-1)
        if labels is not None:
            total_loss = total_loss + contrastive_loss(q, doc, labels, temperature)
        else:
            total_loss = total_loss + infonce_loss(q, doc, temperature)
    return total_loss / len(dims)
