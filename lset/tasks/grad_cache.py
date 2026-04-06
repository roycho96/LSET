"""GradCache: large contrastive batches without OOM."""

from __future__ import annotations

import torch

from lset.losses.fused_contrastive import fused_contrastive_loss
from lset.losses.infonce import infonce_loss
from lset.losses.matryoshka import matryoshka_loss
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.gather import gather_with_grad


def _plan_chunks_token_budget(seq_lengths, budget):
    """Greedy: accumulate sequences until token count exceeds budget.

    Returns list of (begin, end) tuples — sequence index half-open intervals.
    """
    N = len(seq_lengths)
    chunks = []
    begin = 0
    current_tokens = 0
    for i in range(N):
        L = int(seq_lengths[i])
        if current_tokens + L > budget and current_tokens > 0:
            chunks.append((begin, i))
            begin = i
            current_tokens = 0
        current_tokens += L
    if begin < N:
        chunks.append((begin, N))
    return chunks


class GradCacheWrapper:
    """Orchestrates GradCache training — NOT an nn.Module.

    3-step process:
    1. Cache: no_grad encode all sequences → cache embeddings
    2. Loss: Full sim matrix on cached embeddings → get embedding gradients
    3. Replay: Re-encode in chunks → surrogate backward with cached grads
    """

    def __init__(
        self, task: BiEncoderTask, chunk_size: int = 16, token_budget: int | None = None, selective_keep: float = 1.0
    ):
        self.task = task
        self.chunk_size = chunk_size
        self.token_budget = token_budget
        self.selective_keep = selective_keep

    def __call__(
        self, model, query_batch, doc_batch, labels=None, scores=None, pos_qi=None, pos_di=None, pos_counts=None
    ):
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
            from lset.tasks.bi_encoder import _expand_labels_for_gather

            labels = labels.to(q_emb.device)
            labels = _expand_labels_for_gather(labels, q_emb.shape[0], d_emb.shape[0])
            s = None
            if scores is not None:
                s = scores.to(q_emb.device)
                s = _expand_labels_for_gather(s, q_emb.shape[0], d_emb.shape[0], fill=float("-inf"))
            pqi = pos_qi.to(q_emb.device) if pos_qi is not None else None
            pdi = pos_di.to(q_emb.device) if pos_di is not None else None
            pco = pos_counts.to(q_emb.device) if pos_counts is not None else None
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_emb, d_emb, self.task.matryoshka_dims, self.task.temperature, labels)
            else:
                loss = fused_contrastive_loss(
                    q_emb,
                    d_emb,
                    labels,
                    self.task.temperature,
                    s,
                    pos_qi=pqi,
                    pos_di=pdi,
                    pos_counts=pco,
                )
        else:
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_emb, d_emb, self.task.matryoshka_dims, self.task.temperature)
            elif self.task.cascade:
                from lset.losses.cascade_infonce import cascade_infonce_loss

                loss = cascade_infonce_loss(
                    q_emb,
                    d_emb,
                    self.task.temperature,
                    d_small=self.task.cascade_d_small,
                    K_prime=self.task.cascade_K_prime,
                )
            else:
                loss = infonce_loss(q_emb, d_emb, self.task.temperature, self.task.top_k)

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

    @staticmethod
    def _trim_chunk(chunk):
        """Trim padded chunk to actual max length — saves compute on short chunks."""
        if "attention_mask" not in chunk:
            return chunk
        max_len = int(chunk["attention_mask"].sum(dim=1).max().item())
        if max_len < chunk["input_ids"].shape[1]:
            for key in ("input_ids", "attention_mask"):
                if key in chunk:
                    chunk[key] = chunk[key][:, :max_len]
        return chunk

    def _backward_padded_chunks(self, model, batch, grads):
        B = batch["input_ids"].shape[0]

        if self.selective_keep < 1.0:
            # Selective backward: only re-encode samples with large gradient
            grad_norms = grads.norm(dim=1)  # (B,)
            N_keep = max(1, int(B * self.selective_keep))
            important_idx = grad_norms.topk(N_keep).indices.sort().values
            # Re-encode only important samples
            for s in range(0, len(important_idx), self.chunk_size):
                e = min(s + self.chunk_size, len(important_idx))
                idx = important_idx[s:e]
                chunk = {k: v[idx] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                chunk = self._trim_chunk(chunk)
                emb = self.task.encode(model, chunk)
                (emb * grads[idx]).sum().backward()
        else:
            for s in range(0, B, self.chunk_size):
                e = min(s + self.chunk_size, B)
                chunk = {k: v[s:e] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                chunk = self._trim_chunk(chunk)
                emb = self.task.encode(model, chunk)
                (emb * grads[s:e]).sum().backward()

    def _backward_packed_chunks(self, model, batch, grads):
        """Chunk by sequence count or token budget using cu_seqlens."""
        cu = batch["cu_seqlens"]
        num_seqs = cu.shape[0] - 1

        if self.token_budget is not None:
            seq_lengths = (cu[1:] - cu[:-1]).tolist()
            chunk_ranges = _plan_chunks_token_budget(seq_lengths, self.token_budget)
        else:
            chunk_ranges = [(s, min(s + self.chunk_size, num_seqs)) for s in range(0, num_seqs, self.chunk_size)]

        for seq_s, seq_e in chunk_ranges:
            tok_s = int(cu[seq_s])
            tok_e = int(cu[seq_e])
            chunk_cu = cu[seq_s : seq_e + 1] - cu[seq_s]
            chunk = {
                "input_ids": batch["input_ids"][tok_s:tok_e],
                "position_ids": batch["position_ids"][tok_s:tok_e],
                "cu_seqlens": chunk_cu,
                "max_seqlen": int((chunk_cu[1:] - chunk_cu[:-1]).max()),
            }
            emb = self.task.encode(model, chunk)
            (emb * grads[seq_s:seq_e]).sum().backward()
