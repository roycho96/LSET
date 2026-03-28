"""Tests for length-sorted batching in EmbeddingCollator."""

import pytest
from unittest.mock import MagicMock
from lset.data.collator import EmbeddingCollator


def _mock_tokenizer():
    """Create a mock tokenizer that returns input_ids = list of char codes."""
    tok = MagicMock()

    def encode(text):
        result = MagicMock()
        result.ids = list(range(len(text)))  # length = len(text)
        return result

    tok.encode = encode
    return tok


def _make_batch(lengths):
    """Create batch samples with queries of given lengths."""
    return [
        {"query": "x" * l, "positives": ["pos"], "negatives": []}
        for l in lengths
    ]


def test_length_sorted_orders_by_query_length():
    tok = _mock_tokenizer()
    collator = EmbeddingCollator(tok, max_length=512, length_sorted=True)

    batch = _make_batch([10, 50, 20, 40, 30])
    result = collator(batch)

    # With length_sorted, longest queries should come first in the output
    # Query lengths should be monotonically non-increasing
    q_ids = result["query"]["input_ids"]
    q_mask = result["query"]["attention_mask"]
    lengths = q_mask.sum(dim=1).tolist()
    assert lengths == sorted(lengths, reverse=True), f"Not sorted: {lengths}"


def test_length_sorted_reduces_padding():
    tok = _mock_tokenizer()

    # Unsorted batch — high length variance
    batch = _make_batch([10, 100, 20, 90, 15])

    unsorted_collator = EmbeddingCollator(tok, max_length=512, length_sorted=False)
    sorted_collator = EmbeddingCollator(tok, max_length=512, length_sorted=True)

    unsorted_result = unsorted_collator(batch)
    sorted_result = sorted_collator(batch)

    # Both should have the same total real tokens
    unsorted_real = unsorted_result["query"]["attention_mask"].sum().item()
    sorted_real = sorted_result["query"]["attention_mask"].sum().item()
    assert unsorted_real == sorted_real

    # Padding is the same for the full batch (both pad to max)
    # But sorting helps when combined with GradCache chunking


def test_length_sorted_preserves_label_matrix():
    """Length sorting reorders samples but label matrix should still be correct."""
    tok = _mock_tokenizer()
    collator = EmbeddingCollator(tok, max_length=512, length_sorted=True)

    batch = [
        {"query": "short", "positives": ["pos_a"], "negatives": []},
        {"query": "a much longer query", "positives": ["pos_b"], "negatives": []},
    ]
    result = collator(batch)

    # After sorting, longer query comes first
    labels = result["labels"]
    assert labels.shape[0] == 2  # 2 queries
    assert labels.shape[1] == 2  # 2 docs
    # Each query should have exactly 1 positive
    assert labels.sum(dim=1).tolist() == [1.0, 1.0]
    # Diagonal should be positive (sorted: long→pos_b, short→pos_a)
    assert labels[0, 0] == 1.0  # first query (long) -> first doc
    assert labels[1, 1] == 1.0  # second query (short) -> second doc


def test_length_sorted_off_preserves_order():
    tok = _mock_tokenizer()
    collator = EmbeddingCollator(tok, max_length=512, length_sorted=False)

    batch = _make_batch([10, 50, 20])
    result = collator(batch)

    q_mask = result["query"]["attention_mask"]
    lengths = q_mask.sum(dim=1).tolist()
    assert lengths == [10, 50, 20]  # Original order preserved
