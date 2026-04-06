"""Bi-encoder task for contrastive embedding training."""

import torch
import torch.nn as nn

from lset.losses.cascade_infonce import cascade_infonce_loss
from lset.losses.fused_contrastive import fused_contrastive_loss
from lset.losses.infonce import infonce_loss
from lset.losses.matryoshka import matryoshka_loss
from lset.tasks.gather import gather_with_grad
from lset.tasks.packed_pooling import packed_pool
from lset.tasks.pooling import pool


def _expand_labels_for_gather(labels: torch.Tensor, total_q: int, total_d: int, fill: float = 0.0) -> torch.Tensor:
    """Expand local label matrix to match gathered embedding sizes.

    When using gather_with_grad in multi-GPU, embeddings grow but labels stay local.
    This creates a full (total_q, total_d) matrix where the local labels are placed
    in the correct block-diagonal position, and other entries are filled with `fill`.
    """
    local_q, local_d = labels.shape
    if local_q == total_q and local_d == total_d:
        return labels

    import torch.distributed as dist

    if not dist.is_initialized():
        return labels

    rank = dist.get_rank()
    full = torch.full((total_q, total_d), fill, device=labels.device, dtype=labels.dtype)
    q_start = rank * local_q
    d_start = rank * local_d
    full[q_start : q_start + local_q, d_start : d_start + local_d] = labels
    return full


class BiEncoderTask(nn.Module):
    def __init__(
        self,
        pooling: str = "last_token",
        normalize: bool = True,
        temperature: float = 0.02,
        matryoshka_dims: list[int] | None = None,
        top_k: int | None = None,
        cascade: bool = False,
        cascade_d_small: int = 64,
        cascade_K_prime: int = 256,
    ):
        super().__init__()
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.matryoshka_dims = matryoshka_dims
        self.top_k = top_k
        self.cascade = cascade
        self.cascade_d_small = cascade_d_small
        self.cascade_K_prime = cascade_K_prime

    def encode(self, model: nn.Module, batch: dict) -> torch.Tensor:
        if "cu_seqlens" in batch:
            return self._encode_packed(model, batch)
        return self._encode_padded(model, batch)

    def _encode_padded(self, model: nn.Module, batch: dict) -> torch.Tensor:
        out = model(batch["input_ids"], batch["attention_mask"])
        norm = self.normalize and self.matryoshka_dims is None
        return pool(out["hidden_states"], batch["attention_mask"], self.pooling, norm)

    def _encode_packed(self, model: nn.Module, batch: dict) -> torch.Tensor:
        out = model.forward_packed(
            batch["input_ids"],
            batch["position_ids"],
            batch["cu_seqlens"],
            batch["max_seqlen"],
        )
        norm = self.normalize and self.matryoshka_dims is None
        return packed_pool(out["hidden_states"], batch["cu_seqlens"], self.pooling, norm)

    def forward(
        self,
        model: nn.Module,
        query_batch: dict,
        doc_batch: dict,
        neg_batch: dict | None = None,
        labels: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        pos_qi: torch.Tensor | None = None,
        pos_di: torch.Tensor | None = None,
        pos_counts: torch.Tensor | None = None,
    ) -> dict:
        q_emb = self.encode(model, query_batch)
        d_emb = self.encode(model, doc_batch)

        # Gather across GPUs for larger contrastive batch
        q_emb = gather_with_grad(q_emb)
        d_emb = gather_with_grad(d_emb)

        if neg_batch is not None:
            n_emb = self.encode(model, neg_batch)
            n_emb = gather_with_grad(n_emb)
            d_emb = torch.cat([d_emb, n_emb], dim=0)

        if labels is not None:
            # Label-matrix-aware path
            labels = labels.to(q_emb.device)
            labels = _expand_labels_for_gather(labels, q_emb.shape[0], d_emb.shape[0])
            s = None
            if scores is not None:
                s = scores.to(q_emb.device)
                s = _expand_labels_for_gather(s, q_emb.shape[0], d_emb.shape[0], fill=float("-inf"))
            # Move pos_qi/pos_di/pos_counts to device
            pqi = pos_qi.to(q_emb.device) if pos_qi is not None else None
            pdi = pos_di.to(q_emb.device) if pos_di is not None else None
            pco = pos_counts.to(q_emb.device) if pos_counts is not None else None

            if self.matryoshka_dims:
                loss = matryoshka_loss(q_emb, d_emb, self.matryoshka_dims, self.temperature, labels)
            else:
                loss = fused_contrastive_loss(
                    q_emb,
                    d_emb,
                    labels,
                    self.temperature,
                    s,
                    pos_qi=pqi,
                    pos_di=pdi,
                    pos_counts=pco,
                )
        else:
            # Legacy: diagonal positive (backward compatible)
            if self.matryoshka_dims is not None:
                loss = matryoshka_loss(q_emb, d_emb, self.matryoshka_dims, self.temperature)
            elif self.cascade:
                loss = cascade_infonce_loss(
                    q_emb,
                    d_emb,
                    self.temperature,
                    d_small=self.cascade_d_small,
                    K_prime=self.cascade_K_prime,
                )
            else:
                loss = infonce_loss(q_emb, d_emb, self.temperature, self.top_k)

        return {"loss": loss, "query_embeds": q_emb.detach(), "doc_embeds": d_emb.detach()}
