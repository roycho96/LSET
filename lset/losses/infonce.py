"""InfoNCE (NT-Xent) contrastive loss with optional top-K truncation."""

import torch
import torch.nn.functional as F


def infonce_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
                 temperature: float = 0.02,
                 top_k: int | None = None) -> torch.Tensor:
    """Compute InfoNCE loss for contrastive learning.

    Positives are along the diagonal (query[i] matches doc[i]).

    When ``top_k`` is set and ``top_k < B - 1``, uses truncated InfoNCE:
    only the top-K hardest negatives (plus the positive) participate in the
    softmax denominator.  Phase 0 showed gradient cosine > 0.9997 at K=64,
    τ=0.02 — making this a lossless approximation for training.

    Args:
        query_embeds: [B, D] normalized query embeddings.
        doc_embeds: [B, D] normalized document embeddings.
        temperature: Temperature scaling factor.
        top_k: If set, keep only top-K negatives per query.  None or 0 uses
            all negatives (standard InfoNCE).

    Returns:
        Scalar loss tensor.
    """
    sim = torch.matmul(query_embeds, doc_embeds.t()) / temperature  # [B, B]
    B = sim.size(0)

    if top_k and 0 < top_k < B - 1:
        # Truncated InfoNCE: softmax over {positive} ∪ {top-K negatives}
        pos_sim = sim.diagonal().unsqueeze(1)              # (B, 1)
        sim_masked = sim.clone()
        sim_masked.fill_diagonal_(float("-inf"))
        topk_sims, _ = sim_masked.topk(top_k, dim=1)      # (B, K)
        logits = torch.cat([pos_sim, topk_sims], dim=1)    # (B, K+1)
        labels = torch.zeros(B, dtype=torch.long, device=sim.device)
        return F.cross_entropy(logits, labels)

    labels = torch.arange(B, device=sim.device)
    return F.cross_entropy(sim, labels)
