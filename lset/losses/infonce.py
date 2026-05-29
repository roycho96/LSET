"""InfoNCE (NT-Xent) contrastive loss with optional top-K truncation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    temperature: float = 0.02,
    top_k: int | None = None,
) -> torch.Tensor:
    """Diagonal InfoNCE with optional hard-negative truncation."""
    B = query_embeds.shape[0]
    if B == 0:
        return (query_embeds.sum() * 0.0).to(query_embeds.dtype)

    sim = (query_embeds @ doc_embeds.t()) / temperature  # (B, B)

    if top_k and 0 < top_k < B - 1:
        pos_sim = sim.diagonal().unsqueeze(1)            # (B, 1)
        sim_masked = sim.clone()
        sim_masked.fill_diagonal_(float("-inf"))
        topk_sims, _ = sim_masked.topk(top_k, dim=1)     # (B, K)
        logits = torch.cat([pos_sim, topk_sims], dim=1)  # (B, K+1)
        labels = torch.zeros(B, dtype=torch.long, device=sim.device)
        return F.cross_entropy(logits, labels)

    labels = torch.arange(B, device=sim.device)
    return F.cross_entropy(sim, labels)
