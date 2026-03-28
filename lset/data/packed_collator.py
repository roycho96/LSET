"""Packed collator for sequence packing (no padding)."""

import torch
from .packing import pack_sequences


class PackedCollator:
    """Collates samples into packed format (no padding).

    Query sequences packed together, doc sequences packed together.

    Returns:
        {
            "query": {"input_ids": (total_q_tokens,), "cu_seqlens": (num_q+1,), ...},
            "doc": {"input_ids": (total_d_tokens,), "cu_seqlens": (num_d+1,), ...},
        }
    """

    def __call__(self, batch: list[dict]) -> dict:
        queries = [s["query"]["input_ids"] for s in batch]
        positives = [s["positive"]["input_ids"] for s in batch]

        result = {
            "query": pack_sequences(queries),
            "doc": pack_sequences(positives),
        }

        if "negative" in batch[0]:
            negatives = [s["negative"]["input_ids"] for s in batch]
            result["neg"] = pack_sequences(negatives)

        return result


def _pack_fixed_budget(
    token_ids_list: list[list[int]],
    token_budget: int,
    pad_token_id: int,
) -> dict:
    """Pack sequences into a fixed-size 1D tensor.

    Sequences are added greedily until the budget is exhausted. Any sequence
    whose tokens would exceed the remaining budget is skipped. The tail of
    the tensor is filled with ``pad_token_id``; these padding tokens sit
    outside all ``cu_seqlens`` boundaries so flash_attn_varlen_func and
    pooling never see them.

    Returns the same dict layout as :func:`pack_sequences` plus a
    ``"num_sequences"`` key indicating how many sequences were actually
    packed (useful for downstream label slicing).
    """
    all_ids: list[int] = []
    all_positions: list[int] = []
    cu_seqlens: list[int] = [0]
    max_seqlen = 0
    num_packed = 0

    for ids in token_ids_list:
        seq_len = len(ids)
        if seq_len == 0:
            continue
        if cu_seqlens[-1] + seq_len > token_budget:
            # This sequence does not fit — skip it.
            continue
        all_ids.extend(ids)
        all_positions.extend(range(seq_len))
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
        if seq_len > max_seqlen:
            max_seqlen = seq_len
        num_packed += 1

    total_packed = len(all_ids)
    pad_len = token_budget - total_packed

    # Pad input_ids and position_ids to the fixed budget size.
    if pad_len > 0:
        all_ids.extend([pad_token_id] * pad_len)
        all_positions.extend([0] * pad_len)

    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "max_seqlen": max_seqlen if max_seqlen > 0 else 1,
        "position_ids": torch.tensor(all_positions, dtype=torch.long),
        "num_sequences": num_packed,
    }


class FixedBudgetPackedCollator:
    """Packs sequences into a fixed-size tensor for CUDA Graph compatibility.

    CUDA Graphs require static tensor shapes across replays. Standard packed
    collation produces variable-length tensors. This collator guarantees every
    output ``input_ids`` tensor has shape ``(token_budget,)`` by:

    * Greedily packing sequences until the budget is reached.
    * Dropping sequences that do not fit (truncation at sequence granularity).
    * Padding the remainder with ``pad_token_id``.

    The padding tokens sit *after* the last ``cu_seqlens`` boundary, so
    ``flash_attn_varlen_func`` and pooling kernels never attend to or pool
    over them — the extra memory is the only cost.

    Args:
        token_budget: Fixed total token count for each packed tensor.
        pad_token_id: Token id used for padding beyond packed sequences.
    """

    def __init__(self, token_budget: int, pad_token_id: int = 0):
        self.token_budget = token_budget
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        queries = [s["query"]["input_ids"] for s in batch]
        positives = [s["positive"]["input_ids"] for s in batch]

        result = {
            "query": _pack_fixed_budget(queries, self.token_budget, self.pad_token_id),
            "doc": _pack_fixed_budget(positives, self.token_budget, self.pad_token_id),
        }

        if "negative" in batch[0]:
            negatives = [s["negative"]["input_ids"] for s in batch]
            result["neg"] = _pack_fixed_budget(negatives, self.token_budget, self.pad_token_id)

        return result
