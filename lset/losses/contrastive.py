"""Label-matrix-aware contrastive loss supporting multi-positive and soft labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_like(q: torch.Tensor) -> torch.Tensor:
    """Grad-carrying scalar zero on the same device/dtype as ``q``."""
    return (q.sum() * 0.0).to(q.dtype)


def contrastive_loss(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.02,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contrastive loss with a (Q, K) label matrix."""
    sim = (query_embeds @ doc_embeds.T) / temperature  # (Q, K)
    ignore_mask = labels < 0                           # True where labels == -1
    valid_mask = ~ignore_mask

    if scores is not None:
        # Soft-label CE — target is `scores` restricted to non-ignore columns.
        target = scores.to(sim.dtype).masked_fill(ignore_mask, 0.0)
        target_sum = target.sum(dim=-1)                # (Q,)
        valid_query = (target_sum > 0) & valid_mask.any(dim=-1)
        if not valid_query.any():
            return _zero_like(query_embeds)

        target = target / target_sum.clamp(min=1e-12).unsqueeze(-1)  # rows sum to 1
        logits = sim.masked_fill(ignore_mask, float("-inf"))
        log_denom = torch.logsumexp(logits, dim=-1)                  # (Q,)
        per_q = log_denom - (target * sim).sum(dim=-1)               # (Q,)
        return per_q[valid_query].mean()

    # Multi-positive InfoNCE
    pos_mask = labels == 1
    num_pos = pos_mask.sum(dim=-1)
    valid_query = (num_pos > 0) & valid_mask.any(dim=-1)
    if not valid_query.any():
        return _zero_like(query_embeds)

    logits = sim.masked_fill(ignore_mask, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=-1)                       # (Q,)

    pos_sum = (sim * pos_mask.to(sim.dtype)).sum(dim=-1)
    num_pos_safe = num_pos.clamp(min=1).to(sim.dtype)
    per_q = -pos_sum / num_pos_safe + log_denom                       # (Q,)
    return per_q[valid_query].mean()
