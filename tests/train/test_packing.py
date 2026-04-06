"""Tests for sequence packing utilities."""

import torch
from lset.train.data.packing import pack_sequences


def test_pack_sequences_basic():
    """pack_sequences produces correct cu_seqlens and position_ids."""
    seqs = [[10, 20], [30, 40, 50], [60]]
    packed = pack_sequences(seqs)

    assert packed["input_ids"].tolist() == [10, 20, 30, 40, 50, 60]
    assert packed["cu_seqlens"].tolist() == [0, 2, 5, 6]
    assert packed["max_seqlen"] == 3
    assert packed["position_ids"].tolist() == [0, 1, 0, 1, 2, 0]


def test_pack_sequences_total_tokens():
    """Total tokens == sum of sequence lengths."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = pack_sequences(seqs)

    total = sum(len(s) for s in seqs)
    assert packed["input_ids"].shape[0] == total
    assert packed["position_ids"].shape[0] == total
    assert int(packed["cu_seqlens"][-1]) == total


def test_pack_sequences_single():
    """Single sequence packing."""
    packed = pack_sequences([[100, 200, 300]])
    assert packed["cu_seqlens"].tolist() == [0, 3]
    assert packed["position_ids"].tolist() == [0, 1, 2]
    assert packed["max_seqlen"] == 3


def test_pack_sequences_dtypes():
    """Check tensor dtypes."""
    packed = pack_sequences([[1, 2], [3]])
    assert packed["input_ids"].dtype == torch.long
    assert packed["cu_seqlens"].dtype == torch.int32
    assert packed["position_ids"].dtype == torch.long


if __name__ == "__main__":
    test_pack_sequences_basic()
    test_pack_sequences_total_tokens()
    test_pack_sequences_single()
    test_pack_sequences_dtypes()
    print("All packing tests passed!")
