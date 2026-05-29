"""Packed collator for sequence packing (no padding)."""

import torch

from lset.train.data.packing import pack_sequences


class PackedCollator:
    """Collates samples into packed format (no padding)."""

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
    """Pack sequences into a fixed-size 1D tensor."""
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
    """Packs sequences into a fixed-size tensor for CUDA Graph compatibility."""

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
