"""Tests for FixedBudgetPackedCollator."""


from lset.train.data.packed_collator import FixedBudgetPackedCollator, _pack_fixed_budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(query_ids: list[int], positive_ids: list[int], negative_ids=None):
    """Create a sample dict matching the PackedCollator expected input."""
    sample = {
        "query": {"input_ids": query_ids},
        "positive": {"input_ids": positive_ids},
    }
    if negative_ids is not None:
        sample["negative"] = {"input_ids": negative_ids}
    return sample


# ---------------------------------------------------------------------------
# Tests for _pack_fixed_budget (low-level)
# ---------------------------------------------------------------------------

class TestPackFixedBudget:
    def test_output_shape_exact_fit(self):
        """When total tokens == budget, no padding needed."""
        seqs = [[1, 2, 3], [4, 5]]
        result = _pack_fixed_budget(seqs, token_budget=5, pad_token_id=0)
        assert result["input_ids"].shape == (5,)
        assert result["input_ids"].tolist() == [1, 2, 3, 4, 5]
        assert result["cu_seqlens"].tolist() == [0, 3, 5]
        assert result["num_sequences"] == 2

    def test_output_shape_with_padding(self):
        """When total tokens < budget, pad to budget."""
        seqs = [[1, 2], [3]]
        result = _pack_fixed_budget(seqs, token_budget=8, pad_token_id=99)
        assert result["input_ids"].shape == (8,)
        # First 3 tokens are real, rest are pad
        assert result["input_ids"][:3].tolist() == [1, 2, 3]
        assert (result["input_ids"][3:] == 99).all()
        assert result["cu_seqlens"].tolist() == [0, 2, 3]
        assert result["num_sequences"] == 2

    def test_truncation_drops_sequences(self):
        """When total tokens would exceed budget, skip sequences that don't fit."""
        seqs = [[1, 2, 3], [4, 5, 6], [7, 8]]
        # Budget=5: first seq (3) fits, second seq (3) would make 6 > 5, skip.
        # Third seq (2) would make 5 <= 5, fits.
        result = _pack_fixed_budget(seqs, token_budget=5, pad_token_id=0)
        assert result["input_ids"].shape == (5,)
        assert result["input_ids"].tolist() == [1, 2, 3, 7, 8]
        assert result["cu_seqlens"].tolist() == [0, 3, 5]
        assert result["num_sequences"] == 2

    def test_truncation_all_skipped_except_first(self):
        """When only the first sequence fits."""
        seqs = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = _pack_fixed_budget(seqs, token_budget=4, pad_token_id=0)
        assert result["input_ids"].shape == (4,)
        assert result["input_ids"].tolist() == [1, 2, 3, 4]
        assert result["cu_seqlens"].tolist() == [0, 4]
        assert result["num_sequences"] == 1

    def test_no_sequence_fits(self):
        """When budget is too small for any sequence."""
        seqs = [[1, 2, 3], [4, 5, 6]]
        result = _pack_fixed_budget(seqs, token_budget=2, pad_token_id=0)
        assert result["input_ids"].shape == (2,)
        assert (result["input_ids"] == 0).all()
        assert result["cu_seqlens"].tolist() == [0]
        assert result["num_sequences"] == 0

    def test_padding_is_at_end(self):
        """Padding tokens appear only after all packed sequences."""
        seqs = [[10, 20], [30]]
        result = _pack_fixed_budget(seqs, token_budget=6, pad_token_id=0)
        ids = result["input_ids"].tolist()
        last_real = result["cu_seqlens"][-1].item()
        # Everything before last_real is real tokens
        assert ids[:last_real] == [10, 20, 30]
        # Everything from last_real onward is padding
        assert ids[last_real:] == [0, 0, 0]

    def test_cu_seqlens_boundaries(self):
        """cu_seqlens boundaries correctly delimit sequences."""
        seqs = [[1], [2, 3], [4, 5, 6]]
        result = _pack_fixed_budget(seqs, token_budget=10, pad_token_id=0)
        cu = result["cu_seqlens"].tolist()
        ids = result["input_ids"].tolist()
        assert cu == [0, 1, 3, 6]
        assert ids[cu[0]:cu[1]] == [1]
        assert ids[cu[1]:cu[2]] == [2, 3]
        assert ids[cu[2]:cu[3]] == [4, 5, 6]

    def test_position_ids_reset_per_sequence(self):
        """Position IDs reset to 0 at each sequence boundary."""
        seqs = [[10, 20, 30], [40, 50]]
        result = _pack_fixed_budget(seqs, token_budget=8, pad_token_id=0)
        pos = result["position_ids"].tolist()
        assert pos[:3] == [0, 1, 2]
        assert pos[3:5] == [0, 1]

    def test_max_seqlen(self):
        """max_seqlen reflects the longest packed sequence."""
        seqs = [[1], [2, 3, 4], [5, 6]]
        result = _pack_fixed_budget(seqs, token_budget=10, pad_token_id=0)
        assert result["max_seqlen"] == 3

    def test_empty_sequences_skipped(self):
        """Empty sequences in the input list are ignored."""
        seqs = [[], [1, 2], [], [3]]
        result = _pack_fixed_budget(seqs, token_budget=5, pad_token_id=0)
        assert result["num_sequences"] == 2
        assert result["input_ids"][:3].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests for FixedBudgetPackedCollator (high-level)
# ---------------------------------------------------------------------------

class TestFixedBudgetPackedCollator:
    def test_basic_output_shapes(self):
        """Both query and doc tensors have shape (token_budget,)."""
        collator = FixedBudgetPackedCollator(token_budget=16, pad_token_id=0)
        batch = [
            _make_sample([1, 2, 3], [10, 20]),
            _make_sample([4, 5], [30, 40, 50]),
        ]
        result = collator(batch)
        assert result["query"]["input_ids"].shape == (16,)
        assert result["doc"]["input_ids"].shape == (16,)

    def test_negative_key(self):
        """Negative sequences are packed when present."""
        collator = FixedBudgetPackedCollator(token_budget=12, pad_token_id=0)
        batch = [
            _make_sample([1, 2], [10, 20], negative_ids=[100, 200, 300]),
            _make_sample([3], [30], negative_ids=[400]),
        ]
        result = collator(batch)
        assert "neg" in result
        assert result["neg"]["input_ids"].shape == (12,)

    def test_no_negative_key(self):
        """No 'neg' key when samples lack negatives."""
        collator = FixedBudgetPackedCollator(token_budget=12, pad_token_id=0)
        batch = [
            _make_sample([1, 2], [10, 20]),
        ]
        result = collator(batch)
        assert "neg" not in result

    def test_shapes_consistent_across_batches(self):
        """Different batch contents produce the same tensor shapes."""
        collator = FixedBudgetPackedCollator(token_budget=20, pad_token_id=0)

        batch_a = [_make_sample([1, 2], [10, 20, 30])]
        batch_b = [
            _make_sample([1, 2, 3, 4, 5], [10]),
            _make_sample([6, 7], [20, 30, 40, 50]),
        ]

        result_a = collator(batch_a)
        result_b = collator(batch_b)

        assert result_a["query"]["input_ids"].shape == result_b["query"]["input_ids"].shape
        assert result_a["doc"]["input_ids"].shape == result_b["doc"]["input_ids"].shape

    def test_cu_seqlens_last_leq_budget(self):
        """The last cu_seqlens entry never exceeds token_budget."""
        collator = FixedBudgetPackedCollator(token_budget=10, pad_token_id=0)
        batch = [
            _make_sample([1, 2, 3], [10, 20]),
            _make_sample([4, 5], [30, 40, 50, 60]),
        ]
        result = collator(batch)
        for key in ("query", "doc"):
            last_cu = result[key]["cu_seqlens"][-1].item()
            assert last_cu <= 10, f"{key} cu_seqlens[-1]={last_cu} > budget=10"
