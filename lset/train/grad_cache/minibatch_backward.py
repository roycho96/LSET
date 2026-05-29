"""Minibatch planning + runtime-specific backward policy for GradCache."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel as nn_parallel

from torch import Tensor


def _num_units(sentence_feature: dict) -> int:
    if "cu_seqlens" in sentence_feature:
        return int(sentence_feature["cu_seqlens"].shape[0]) - 1
    return int(sentence_feature["input_ids"].shape[0])


def _seq_lengths(sentence_feature: dict) -> Tensor:
    cu = sentence_feature["cu_seqlens"]
    if "seq_lengths" in sentence_feature:
        return sentence_feature["seq_lengths"]
    return cu[1:] - cu[:-1]


def _by_fixed_size(n: int, mini_batch_size: int) -> list[tuple[int, int]]:
    chunks = []
    for b in range(0, n, mini_batch_size):
        e = min(b + mini_batch_size, n)
        chunks.append((b, e))
    return chunks


def _by_token_budget(seq_lengths, token_budget: int) -> list[tuple[int, int]]:
    """Greedy — accumulate sequences until adding the next would exceed budget."""
    chunks = []
    begin = 0
    current = 0
    N = len(seq_lengths)
    for i in range(N):
        L = int(seq_lengths[i])
        if current + L > token_budget and current > 0:
            chunks.append((begin, i))
            begin = i
            current = 0
        current += L
    if begin < N:
        chunks.append((begin, N))
    return chunks


def plan_minibatches(
    sentence_feature: dict,
    mini_batch_size: int,
    token_budget: int | None = None,
) -> list[tuple[int, int]]:
    """Split a padded or packed feature dict into ``[(begin, end), ...]``."""
    if "cu_seqlens" in sentence_feature:
        if token_budget is not None:
            return _by_token_budget(_seq_lengths(sentence_feature), token_budget)
        return _by_fixed_size(_num_units(sentence_feature), mini_batch_size)
    return _by_fixed_size(_num_units(sentence_feature), mini_batch_size)


def _is_deepspeed_engine(m) -> bool:
    """Duck-type check — DS versions vary, attribute presence is stable."""
    return (
        hasattr(m, "set_gradient_accumulation_boundary")
        and hasattr(m, "backward")
        and hasattr(m, "module")
    )


def _is_fsdp2_module(m) -> bool:
    """FSDP2 (torch.distributed._composable.fsdp) exposes set_requires_gradient_sync."""
    return hasattr(m, "set_requires_gradient_sync") and hasattr(m, "set_reshard_after_backward")


class MinibatchBackward:
    """Runtime-specific context + per-minibatch backward dispatch."""

    @classmethod
    def for_model(cls, forward_model, *, is_ga_boundary: bool = True) -> "MinibatchBackward":
        if _is_deepspeed_engine(forward_model):
            return DeepSpeedMinibatchBackward(forward_model, is_ga_boundary=is_ga_boundary)
        if isinstance(forward_model, nn_parallel.DistributedDataParallel):
            return DDPMinibatchBackward(forward_model, is_ga_boundary=is_ga_boundary)
        if _is_fsdp2_module(forward_model):
            return FSDP2MinibatchBackward(forward_model, is_ga_boundary=is_ga_boundary)
        return BasicMinibatchBackward()

    def context(self):
        """Context manager wrapping the whole minibatch replay loop."""
        return nullcontext()

    def backward(self, surrogate: Tensor, is_last: bool) -> None:
        raise NotImplementedError

    def finalize(self, params: Iterable[nn.Parameter]) -> None:
        """Called once after all minibatches; post-replay grad sync if needed."""
        return None

    def align_minibatches(
        self,
        minibatches_per_feature: list[list[tuple[int, int]]],
    ) -> tuple[list[list[tuple[int, int]]], list[int]]:
        """Pad each feature's per-rank plan to the world-max via all_reduce(MAX)."""
        num_valid = [len(mbs) for mbs in minibatches_per_feature]
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return [list(mbs) for mbs in minibatches_per_feature], num_valid

        dev = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        max_counts = torch.tensor(num_valid, device=dev, dtype=torch.long)
        dist.all_reduce(max_counts, op=dist.ReduceOp.MAX)
        targets = max_counts.tolist()

        aligned: list[list[tuple[int, int]]] = []
        for mbs, n_target, n_local in zip(minibatches_per_feature, targets, num_valid):
            if n_target > n_local:
                last = mbs[-1] if n_local > 0 else (0, 0)
                aligned.append(list(mbs) + [last] * (n_target - n_local))
            else:
                aligned.append(list(mbs))
        return aligned, num_valid


class BasicMinibatchBackward(MinibatchBackward):
    def backward(self, surrogate: Tensor, is_last: bool) -> None:
        surrogate.backward()


class DDPMinibatchBackward(MinibatchBackward):
    """Suppress DDP reducer during replay, manual bucketed all-reduce on finalize."""

    def __init__(self, ddp_model: nn.Module, is_ga_boundary: bool):
        self.ddp_model = ddp_model
        self.is_ga_boundary = is_ga_boundary

    def context(self):
        ns = getattr(self.ddp_model, "no_sync", None)
        return ns() if callable(ns) else nullcontext()

    def backward(self, surrogate: Tensor, is_last: bool) -> None:
        surrogate.backward()

    def finalize(self, params: Iterable[nn.Parameter]) -> None:
        if not self.is_ga_boundary:
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        grads: list[Tensor] = []
        for p in params:
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        if not grads:
            return
        by_dtype: dict[torch.dtype, list[Tensor]] = defaultdict(list)
        for g in grads:
            by_dtype[g.dtype].append(g)
        for dtype, gs in by_dtype.items():
            flat = torch.cat([g.contiguous().view(-1) for g in gs])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for g in gs:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n


class FSDP2MinibatchBackward(MinibatchBackward):
    """FSDP2 — accumulate with sync off, flip on for the last minibatch only."""

    def __init__(self, fsdp_module: nn.Module, is_ga_boundary: bool):
        self.fsdp_module = fsdp_module
        self.is_ga_boundary = is_ga_boundary
        self._prev_sync: bool | None = None

    def context(self):
        parent = self

        class _Ctx:
            def __enter__(self_):
                parent._prev_sync = True
                try:
                    parent.fsdp_module.set_requires_gradient_sync(False)
                except Exception:
                    pass
                return None

            def __exit__(self_, exc_type, exc_val, exc_tb):
                try:
                    parent.fsdp_module.set_requires_gradient_sync(True)
                except Exception:
                    pass
                return None

        return _Ctx()

    def backward(self, surrogate: Tensor, is_last: bool) -> None:
        if is_last and self.is_ga_boundary:
            try:
                self.fsdp_module.set_requires_gradient_sync(True)
            except Exception:
                pass
        surrogate.backward()


class DeepSpeedMinibatchBackward(MinibatchBackward):
    """DeepSpeed — ``set_gradient_accumulation_boundary`` + ``engine.backward``."""

    def __init__(self, engine, is_ga_boundary: bool = True):
        self.engine = engine
        self.is_ga_boundary = is_ga_boundary

    def backward(self, surrogate: Tensor, is_last: bool) -> None:
        boundary = is_last and self.is_ga_boundary
        self.engine.set_gradient_accumulation_boundary(boundary)
        self.engine.backward(surrogate)
