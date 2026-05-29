"""Token-budget-aware chunked encoder for long-document MTEB evaluation."""

from __future__ import annotations

import hashlib
import pickle

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def cached_tokenize(
    tokenize_fn: Callable[[list[str]], list[list[int]]],
    texts: list[str],
    cache_dir: Path | None = None,
) -> list[list[int]]:
    """Tokenize with optional disk cache keyed on the input hash."""
    if cache_dir is None:
        return tokenize_fn(texts)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Hash first/last chars of each text rather than the full content.
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts:
        s = t[:64] + t[-64:] if len(t) > 128 else t
        h.update(s.encode())
        h.update(b"\0")
    key = h.hexdigest()[:32]
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    tokens = tokenize_fn(texts)
    with path.open("wb") as f:
        pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokens


# Defaults suitable for a 16 GB card with Qwen3 2B bf16. Override per model.
DEFAULT_CHUNK_LEN = 4096
DEFAULT_OVERLAP = 128
DEFAULT_TOKEN_BUDGET = 16384
CPU_STREAM_THRESHOLD = 50_000  # above this N, stream output via np.memmap


def _chunk(ids: list[int], chunk_len: int, overlap: int) -> list[list[int]]:
    """Split ``ids`` into chunks of length ``chunk_len`` with ``overlap``."""
    if len(ids) <= chunk_len:
        return [ids]
    out = []
    step = max(1, chunk_len - overlap)
    for start in range(0, len(ids), step):
        out.append(ids[start : start + chunk_len])
        if start + chunk_len >= len(ids):
            break
    return out


def _plan_batches(seq_lens: list[int], budget: int) -> list[list[int]]:
    """Greedy packer: close a batch when (B+1)·max_len exceeds ``budget``."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for i, L in enumerate(seq_lens):
        proj = (len(current) + 1) * max(current_max, L)
        if current and proj > budget:
            batches.append(current)
            current = []
            current_max = 0
        current.append(i)
        current_max = max(current_max, L)
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def encode_chunked(
    model: torch.nn.Module,
    tokenize_fn: Callable[[list[str]], list[list[int]]],
    pool_fn: Callable[[torch.Tensor, torch.Tensor, str, bool], torch.Tensor],
    texts: list[str],
    *,
    device: torch.device,
    pad_id: int,
    pooling: str = "mean",
    normalize: bool = True,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    overlap: int = DEFAULT_OVERLAP,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_pos: int | None = None,
    out_path: str | None = None,
    dim: int | None = None,
    tokenize_cache_dir: str | None = None,
    oom_retry_halving: bool = True,
) -> np.ndarray:
    """Encode ``texts`` with token-budget batching and per-doc chunking."""
    if pooling not in ("mean", "last_token", "cls"):
        raise ValueError(f"unknown pooling {pooling!r}")
    if pooling != "mean" and any(len(t) > chunk_len for t in texts):
        # last_token/cls pool only makes sense for a single-chunk doc.
        raise ValueError(
            f"pooling={pooling!r} is not defined across chunks; use mean "
            f"pooling for docs longer than chunk_len={chunk_len}"
        )

    # 1. Tokenize (optionally from disk cache) and apply model-level truncation.
    cache_dir = Path(tokenize_cache_dir) if tokenize_cache_dir else None
    encoded = cached_tokenize(tokenize_fn, texts, cache_dir=cache_dir)
    if max_pos is not None:
        encoded = [ids[:max_pos] for ids in encoded]

    # 2. Chunk long docs. Track which doc each chunk belongs to and its len.
    flat_chunks: list[list[int]] = []
    owners: list[int] = []
    chunk_lens: list[int] = []
    for doc_idx, ids in enumerate(encoded):
        for c in _chunk(ids, chunk_len, overlap):
            flat_chunks.append(c)
            owners.append(doc_idx)
            chunk_lens.append(len(c))

    # 3. Plan token-budget batches over the flat chunk list.
    batches = _plan_batches(chunk_lens, token_budget)

    # 4. Output buffer: memmap if streaming to disk, else GPU accumulator.
    N = len(texts)
    if N == 0:
        return np.zeros((0, dim or 0), dtype=np.float32)
    if out_path and N >= CPU_STREAM_THRESHOLD:
        if dim is None:
            raise ValueError("dim required for memmap output path")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        final_out = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(N, dim))
    else:
        final_out = None  # decide dim from first forward

    # Per-doc accumulators (GPU). Weighted sum of chunk embeddings.
    doc_sum: torch.Tensor | None = None
    doc_weight = torch.zeros(N, device=device)

    model.eval()

    def _run_batch(batch_idx: list[int]) -> torch.Tensor:
        Lmax = max(chunk_lens[i] for i in batch_idx)
        B = len(batch_idx)
        ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        for row, i in enumerate(batch_idx):
            c = flat_chunks[i]
            ids[row, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)
            mask[row, : len(c)] = 1
        hidden = model(ids, mask)["hidden_states"]
        return pool_fn(hidden, mask, pooling, False)

    for batch_idx in batches:
        try:
            v = _run_batch(batch_idx)
        except torch.cuda.OutOfMemoryError:
            if not oom_retry_halving or len(batch_idx) == 1:
                raise
            # Halve batch and retry — typically enough to recover.
            torch.cuda.empty_cache()
            mid = len(batch_idx) // 2
            left = _run_batch(batch_idx[:mid])
            right = _run_batch(batch_idx[mid:])
            v = torch.cat([left, right], dim=0)

        if doc_sum is None:
            D = v.shape[-1]
            doc_sum = torch.zeros(N, D, device=device, dtype=torch.float32)

        for row, i in enumerate(batch_idx):
            w = float(chunk_lens[i])
            doc_sum[owners[i]] += v[row].float() * w
            doc_weight[owners[i]] += w

    assert doc_sum is not None  # batches was non-empty for N > 0
    final = doc_sum / doc_weight.clamp_min(1.0).unsqueeze(-1)
    if normalize:
        final = F.normalize(final, p=2, dim=-1)

    if final_out is not None:
        for s in range(0, N, 8192):
            final_out[s : s + 8192] = final[s : s + 8192].cpu().numpy()
        final_out.flush()
        return final_out
    return final.cpu().numpy()
