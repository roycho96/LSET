"""Pack variable-length sequences into contiguous tensors."""

import torch


def pack_sequences(token_ids_list: list[list[int]]) -> dict:
    """Pack variable-length sequences into a single 1D tensor."""
    all_ids = []
    all_positions = []
    cu_seqlens = [0]
    for ids in token_ids_list:
        all_ids.extend(ids)
        all_positions.extend(range(len(ids)))
        cu_seqlens.append(cu_seqlens[-1] + len(ids))

    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "max_seqlen": max(len(ids) for ids in token_ids_list),
        "position_ids": torch.tensor(all_positions, dtype=torch.long),
    }
