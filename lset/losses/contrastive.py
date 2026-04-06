"""Label-matrix-aware contrastive loss supporting multi-positive and soft labels."""

import torch
import torch.nn.functional as F


def contrastive_loss(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.02,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contrastive loss with label matrix.

    Args:
        query_embeds: (Q, D) normalized query embeddings.
        doc_embeds: (K, D) normalized doc embeddings.
        labels: (Q, K) — 1=positive, 0=negative/in-batch-neg, -1=ignore.
        temperature: Scaling temperature.
        scores: (Q, K) optional soft target scores for distillation.

    Returns:
        Scalar loss.
    """
    sim = query_embeds @ doc_embeds.T / temperature  # (Q, K)

    if scores is not None:
        # Soft label cross-entropy
        mask = labels >= 0
        # Replace -inf scores with very negative value for softmax
        safe_scores = scores.clone()
        safe_scores[~mask] = float("-inf")
        target_dist = F.softmax(safe_scores / temperature, dim=-1)
        log_probs = F.log_softmax(sim.masked_fill(~mask, float("-inf")), dim=-1)
        loss = -(target_dist * log_probs * mask.float()).sum(dim=-1)
        loss = loss / mask.float().sum(dim=-1).clamp(min=1)
        return loss.mean()

    # Multi-positive InfoNCE
    pos_mask = (labels == 1).float()
    neg_mask = (labels >= 0).float()  # pos + neg, exclude ignore

    # Log-sum-exp over all non-ignored docs (denominator)
    sim_masked = sim.masked_fill(neg_mask == 0, float("-inf"))
    log_denom = torch.logsumexp(sim_masked, dim=-1)  # (Q,)

    # Average positive similarity per query (numerator)
    num_pos = pos_mask.sum(dim=-1).clamp(min=1)
    pos_sum = (sim * pos_mask).sum(dim=-1)

    loss = -pos_sum / num_pos + log_denom
    return loss.mean()
