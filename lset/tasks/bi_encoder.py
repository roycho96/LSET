"""Bi-encoder task for contrastive embedding training."""

import torch
import torch.nn as nn

from .pooling import pool
from .packed_pooling import packed_pool
from .gather import gather_with_grad
from .losses.infonce import infonce_loss
from .losses.matryoshka import matryoshka_loss


class BiEncoderTask(nn.Module):
    def __init__(self, pooling: str = "last_token", normalize: bool = True,
                 temperature: float = 0.02, matryoshka_dims: list[int] | None = None):
        super().__init__()
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.matryoshka_dims = matryoshka_dims

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
            batch["input_ids"], batch["position_ids"],
            batch["cu_seqlens"], batch["max_seqlen"],
        )
        norm = self.normalize and self.matryoshka_dims is None
        return packed_pool(out["hidden_states"], batch["cu_seqlens"], self.pooling, norm)

    def forward(self, model: nn.Module, query_batch: dict, doc_batch: dict,
                neg_batch: dict | None = None) -> dict:
        q_emb = self.encode(model, query_batch)
        d_emb = self.encode(model, doc_batch)

        # Gather across GPUs for larger contrastive batch
        q_emb = gather_with_grad(q_emb)
        d_emb = gather_with_grad(d_emb)

        if neg_batch is not None:
            n_emb = self.encode(model, neg_batch)
            n_emb = gather_with_grad(n_emb)
            d_emb = torch.cat([d_emb, n_emb], dim=0)

        if self.matryoshka_dims is not None:
            loss = matryoshka_loss(q_emb, d_emb, self.matryoshka_dims, self.temperature)
        else:
            loss = infonce_loss(q_emb, d_emb, self.temperature)

        return {"loss": loss, "query_embeds": q_emb.detach(), "doc_embeds": d_emb.detach()}
