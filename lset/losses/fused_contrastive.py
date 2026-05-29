"""Fused contrastive loss with automatic dispatch."""

import torch

from torch import Tensor

from lset.kernels.loss import fused_dense_loss
from lset.kernels.loss import should_use_fused
from lset.losses.contrastive import contrastive_loss

# Lazy SM100 detection
_sm100_checked = False
_sm100_ok = False


def fused_contrastive_loss(
    query_embeds: Tensor,
    doc_embeds: Tensor,
    labels: Tensor,
    temperature: float = 0.02,
    scores: Tensor | None = None,
    pos_qi: Tensor | None = None,
    pos_di: Tensor | None = None,
    pos_counts: Tensor | None = None,
    loss_type: str = "multi",
) -> Tensor:
    """Contrastive loss with automatic fused kernel dispatch."""
    Q, K = query_embeds.shape[0], doc_embeds.shape[0]

    use_fused = (
        should_use_fused(Q, K, loss_type)
        and query_embeds.is_cuda
        and pos_qi is not None
        and scores is None  # Fused kernel handles soft CE via int8 labels, not score matrix
    )

    if use_fused:
        scale = 1.0 / temperature
        # Convert float labels to int8 for the kernel
        labels_int8 = labels.to(torch.int8)

        # SM100 CUDA kernel exists but is currently slower than Triton
        # (WMMA m16n16k16 < Triton's auto-selected MMA; needs tcgen05.mma
        #  128x256x16 + TMA + TMEM to beat Triton on SM100).
        # Dispatch disabled until SM100-native instructions are implemented.
        # To test manually: lset.kernels.experimental.loss_sm100.fused_dense_loss_sm100()

        return fused_dense_loss(
            query_embeds,
            doc_embeds,
            labels_int8,
            scale=scale,
            loss_type=loss_type,
            pos_qi=pos_qi,
            pos_di=pos_di,
            pos_counts=pos_counts,
        )
    else:
        return contrastive_loss(query_embeds, doc_embeds, labels, temperature, scores)
