"""GradCache — scale contrastive batches beyond the encoder's memory."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from torch import Tensor

from lset.losses.fused_contrastive import fused_contrastive_loss
from lset.losses.infonce import infonce_loss
from lset.losses.matryoshka import matryoshka_loss
from lset.distributed.gather import gather_with_grad
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.bi_encoder import _expand_labels_for_gather
from lset.train.grad_cache.minibatch_backward import MinibatchBackward
from lset.train.grad_cache.minibatch_backward import plan_minibatches
from lset.train.grad_cache.rand_context import RandContext


@dataclass
class _FeatureState:
    sentence_feature: dict
    random_states: list[RandContext | None]
    grads: list[Tensor | None]
    minibatches: list[tuple[int, int]]
    num_valid: int


def _trim_padded(chunk: dict) -> dict:
    """Trim pad tokens inside a minibatch (right-pad only — no-op otherwise)."""
    if "attention_mask" not in chunk or chunk["attention_mask"].dim() != 2:
        return chunk
    max_len = int(chunk["attention_mask"].sum(dim=1).max().item())
    if max_len >= chunk["input_ids"].shape[1]:
        return chunk
    out = dict(chunk)
    for k, v in chunk.items():
        if isinstance(v, Tensor) and v.dim() == 2 and v.shape[1] == chunk["input_ids"].shape[1]:
            out[k] = v[:, :max_len]
    return out


def _slice_padded(feature: dict, begin: int, end: int) -> dict:
    out = {k: (v[begin:end] if isinstance(v, Tensor) else v) for k, v in feature.items()}
    return _trim_padded(out)


def _slice_packed(feature: dict, seq_begin: int, seq_end: int) -> dict:
    cu = feature["cu_seqlens"]
    tok_begin = int(cu[seq_begin])
    tok_end = int(cu[seq_end])
    chunk_cu = cu[seq_begin : seq_end + 1] - cu[seq_begin]
    max_seqlen = int((chunk_cu[1:] - chunk_cu[:-1]).max()) if seq_end > seq_begin else 0
    return {
        "input_ids": feature["input_ids"][tok_begin:tok_end],
        "position_ids": feature["position_ids"][tok_begin:tok_end],
        "cu_seqlens": chunk_cu,
        "max_seqlen": max_seqlen,
    }


def _slice_feature(feature: dict, begin: int, end: int) -> dict:
    if "cu_seqlens" in feature:
        return _slice_packed(feature, begin, end)
    return _slice_padded(feature, begin, end)


class GradCacheWrapper:
    """Encoder-agnostic GradCache driver."""

    def __init__(
        self,
        task: BiEncoderTask,
        chunk_size: int = 16,
        token_budget: int | None = None,
        selective_keep: float = 1.0,
    ):
        self.task = task
        self.chunk_size = chunk_size
        self.token_budget = token_budget
        self.selective_keep = selective_keep

    # --- Step 1 ---------------------------------------------------------------

    def _embed_no_grad(
        self,
        model: nn.Module,
        feature: dict,
        plan: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[RandContext | None]]:
        embeddings: list[Tensor] = []
        rands: list[RandContext | None] = []
        for begin, end in plan:
            chunk = _slice_feature(feature, begin, end)
            tensor_vals = [v for v in chunk.values() if isinstance(v, Tensor)]
            rs = RandContext(*tensor_vals)
            with torch.no_grad():
                with rs:
                    emb = self.task.encode(model, chunk)
            embeddings.append(emb.detach().requires_grad_())
            rands.append(rs)
        return embeddings, rands

    # --- Step 2 ---------------------------------------------------------------

    def _compute_loss_and_grads(
        self,
        q_leaves: list[Tensor],
        d_leaves: list[Tensor],
        num_valid_q: int,
        num_valid_d: int,
        labels: Tensor | None,
        scores: Tensor | None,
        pos_qi: Tensor | None,
        pos_di: Tensor | None,
        pos_counts: Tensor | None,
    ) -> tuple[Tensor, list[Tensor | None], list[Tensor | None], Tensor | None]:
        # Drop dummy leaves (padding from align_minibatches) before the loss.
        q_valid = q_leaves[:num_valid_q]
        d_valid = d_leaves[:num_valid_d]

        q_local = torch.cat(q_valid, dim=0) if q_valid else q_leaves[0].new_empty(0)
        d_local = torch.cat(d_valid, dim=0) if d_valid else d_leaves[0].new_empty(0)

        q_global = gather_with_grad(q_local)
        d_global = gather_with_grad(d_local)

        temperature = self.task.temperature
        logit_scale_param = self._logit_scale_param()

        if labels is not None:
            labels = labels.to(q_global.device)
            labels = _expand_labels_for_gather(labels, q_global.shape[0], d_global.shape[0])
            s = None
            if scores is not None:
                s = scores.to(q_global.device)
                s = _expand_labels_for_gather(s, q_global.shape[0], d_global.shape[0], fill=float("-inf"))
            pqi = pos_qi.to(q_global.device) if pos_qi is not None else None
            pdi = pos_di.to(q_global.device) if pos_di is not None else None
            pco = pos_counts.to(q_global.device) if pos_counts is not None else None
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_global, d_global, self.task.matryoshka_dims, temperature, labels)
            else:
                loss = fused_contrastive_loss(
                    q_global,
                    d_global,
                    labels,
                    temperature,
                    s,
                    pos_qi=pqi,
                    pos_di=pdi,
                    pos_counts=pco,
                )
        else:
            if self.task.matryoshka_dims:
                loss = matryoshka_loss(q_global, d_global, self.task.matryoshka_dims, temperature)
            else:
                loss = infonce_loss(q_global, d_global, temperature, self.task.top_k)

        # Anything with requires_grad_ is a valid grad target. Dummies are None.
        grad_inputs: list[Tensor] = list(q_valid) + list(d_valid)
        want_scale_grad = (
            logit_scale_param is not None
            and isinstance(logit_scale_param, nn.Parameter)
            and logit_scale_param.requires_grad
        )
        if want_scale_grad:
            grad_inputs.append(logit_scale_param)

        if not grad_inputs:
            return loss.detach(), [None] * len(q_leaves), [None] * len(d_leaves), None

        grads = torch.autograd.grad(
            loss,
            grad_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        qn = len(q_valid)
        dn = len(d_valid)
        q_grads: list[Tensor | None] = list(grads[:qn])
        d_grads: list[Tensor | None] = list(grads[qn : qn + dn])
        scale_grad = grads[qn + dn] if want_scale_grad else None

        # Pad to full plan length so Step 3 can iterate the aligned plan;
        # None slots trigger zero-grad surrogates on the dummy minibatches.
        q_grads.extend([None] * (len(q_leaves) - qn))
        d_grads.extend([None] * (len(d_leaves) - dn))
        return loss.detach(), q_grads, d_grads, scale_grad

    # --- Step 3 ---------------------------------------------------------------

    def _replay_and_backward(
        self,
        model: nn.Module,
        feature_states: list[_FeatureState],
        scale_grad: Tensor | None,
        runtime: MinibatchBackward,
        grad_output: float,
    ) -> None:
        total_calls = sum(len(fs.grads) for fs in feature_states) + (1 if scale_grad is not None else 0)
        call_idx = 0

        with runtime.context():
            with torch.enable_grad():
                for fs in feature_states:
                    for (begin, end), rs, grad in zip(fs.minibatches, fs.random_states, fs.grads):
                        chunk = _slice_feature(fs.sentence_feature, begin, end)
                        with (rs if rs is not None else nullcontext()):
                            emb = self.task.encode(model, chunk)
                        if grad is None:
                            grad_tensor = torch.zeros_like(emb)
                        else:
                            grad_tensor = grad.detach().to(emb.dtype)
                        is_last = call_idx == total_calls - 1
                        surrogate = torch.dot(emb.flatten(), grad_tensor.flatten()) * grad_output
                        runtime.backward(surrogate, is_last=is_last)
                        call_idx += 1

                if scale_grad is not None:
                    logit_scale_param = self._logit_scale_param()
                    g = scale_grad.detach().to(logit_scale_param.dtype)
                    surrogate = (logit_scale_param * g).sum() * grad_output
                    is_last = call_idx == total_calls - 1
                    runtime.backward(surrogate, is_last=is_last)
                    call_idx += 1

        runtime.finalize(list(model.parameters()))

    # --- Helpers --------------------------------------------------------------

    def _logit_scale_param(self) -> Tensor | None:
        logit_scale = getattr(self.task, "logit_scale", None)
        if logit_scale is None:
            return None
        # ``LogitScale`` exposes a ``log_scale`` ``nn.Parameter``
        return getattr(logit_scale, "log_scale", None)

    # --- Public entry point ---------------------------------------------------

    def __call__(
        self,
        model: nn.Module,
        query_batch: dict,
        doc_batch: dict,
        labels: Tensor | None = None,
        scores: Tensor | None = None,
        pos_qi: Tensor | None = None,
        pos_di: Tensor | None = None,
        pos_counts: Tensor | None = None,
        is_ga_boundary: bool = True,
        grad_output: float = 1.0,
    ) -> Tensor:
        runtime = MinibatchBackward.for_model(model, is_ga_boundary=is_ga_boundary)

        local_plans = [
            plan_minibatches(query_batch, mini_batch_size=self.chunk_size, token_budget=self.token_budget),
            plan_minibatches(doc_batch, mini_batch_size=self.chunk_size, token_budget=self.token_budget),
        ]
        aligned, num_valid = runtime.align_minibatches(local_plans)
        q_plan, d_plan = aligned
        num_valid_q, num_valid_d = num_valid

        q_leaves, q_rands = self._embed_no_grad(model, query_batch, q_plan)
        d_leaves, d_rands = self._embed_no_grad(model, doc_batch, d_plan)

        loss, q_grads, d_grads, scale_grad = self._compute_loss_and_grads(
            q_leaves=q_leaves,
            d_leaves=d_leaves,
            num_valid_q=num_valid_q,
            num_valid_d=num_valid_d,
            labels=labels,
            scores=scores,
            pos_qi=pos_qi,
            pos_di=pos_di,
            pos_counts=pos_counts,
        )

        feature_states = [
            _FeatureState(
                sentence_feature=query_batch,
                random_states=q_rands,
                grads=q_grads,
                minibatches=q_plan,
                num_valid=num_valid_q,
            ),
            _FeatureState(
                sentence_feature=doc_batch,
                random_states=d_rands,
                grads=d_grads,
                minibatches=d_plan,
                num_valid=num_valid_d,
            ),
        ]

        self._replay_and_backward(
            model=model,
            feature_states=feature_states,
            scale_grad=scale_grad,
            runtime=runtime,
            grad_output=grad_output,
        )
        return loss

    # Back-compat: keep the public helpers referenced by existing tests.
    @staticmethod
    def _trim_chunk(chunk):
        return _trim_padded(chunk)
