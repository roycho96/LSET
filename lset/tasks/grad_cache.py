"""GradCache: large contrastive batches without OOM."""

from __future__ import annotations

import torch

from .bi_encoder import BiEncoderTask
from .gather import gather_with_grad
from .losses.infonce import infonce_loss
from .losses.contrastive import contrastive_loss
from .losses.matryoshka import matryoshka_loss


class GradCacheWrapper:
    """Orchestrates GradCache training — NOT an nn.Module.

    3-step process:
    1. Cache: no_grad encode all sequences → cache embeddings
    2. Loss: Full sim matrix on cached embeddings → get embedding gradients
    3. Replay: Re-encode in chunks → surrogate backward with cached grads
    """

    def __init__(self, task: BiEncoderTask, chunk_size: int = 16):
        self.task = task
        self.chunk_size = chunk_size

    def __call__(self, model, query_batch, doc_batch, labels=None, scores=None):
        # Step 1: no_grad encode
        with torch.no_grad():
            q_emb = self.task.encode(model, query_batch)
            d_emb = self.task.encode(model, doc_batch)

        # Gather across GPUs
        q_emb = gather_with_grad(q_emb)
        d_emb = gather_with_grad(d_emb)

        # Step 2: loss on cached embeddings → get embedding grads
        q_emb = q_emb.detach().requires_grad_(True)
        d_emb = d_emb.detach().requires_grad_(True)

        if labels is not None:
            from .bi_encoder import _expand_labels_for_gather
            labels = labels.to(q_emb.device)
            labels = _expand_labels_for_gather(labels, q_emb.shape[0], d_emb.shape[0])
            s = None
            if scores is not None:
                s = scores.to(q_emb.device)
                s = _expand_labels_for_gather(s, q_emb.shape[0], d_emb.shape[0],
                                               fill=float("-inf"))
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_emb, d_emb, self.task.matryoshka_dims,
                                       self.task.temperature, labels)
            else:
                loss = contrastive_loss(q_emb, d_emb, labels, self.task.temperature, s)
        else:
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_emb, d_emb, self.task.matryoshka_dims,
                                       self.task.temperature)
            else:
                loss = infonce_loss(q_emb, d_emb, self.task.temperature)

        loss.backward()
        q_grad = q_emb.grad.clone()
        d_grad = d_emb.grad.clone()

        # Step 3: chunk re-encode + surrogate backward
        self._backward_chunks(model, query_batch, q_grad)
        self._backward_chunks(model, doc_batch, d_grad)

        return loss.detach()

    def _backward_chunks(self, model, batch, emb_grads):
        if "cu_seqlens" in batch:
            self._backward_packed_chunks(model, batch, emb_grads)
        else:
            self._backward_padded_chunks(model, batch, emb_grads)

    def _backward_padded_chunks(self, model, batch, grads):
        B = batch["input_ids"].shape[0]
        for s in range(0, B, self.chunk_size):
            e = min(s + self.chunk_size, B)
            chunk = {k: v[s:e] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            emb = self.task.encode(model, chunk)
            (emb * grads[s:e]).sum().backward()

    def _backward_packed_chunks(self, model, batch, grads):
        """Chunk by sequence count using cu_seqlens."""
        cu = batch["cu_seqlens"]
        num_seqs = cu.shape[0] - 1
        for seq_s in range(0, num_seqs, self.chunk_size):
            seq_e = min(seq_s + self.chunk_size, num_seqs)
            tok_s = int(cu[seq_s])
            tok_e = int(cu[seq_e])
            chunk_cu = cu[seq_s:seq_e + 1] - cu[seq_s]
            chunk = {
                "input_ids": batch["input_ids"][tok_s:tok_e],
                "position_ids": batch["position_ids"][tok_s:tok_e],
                "cu_seqlens": chunk_cu,
                "max_seqlen": int((chunk_cu[1:] - chunk_cu[:-1]).max()),
            }
            emb = self.task.encode(model, chunk)
            (emb * grads[seq_s:seq_e]).sum().backward()
