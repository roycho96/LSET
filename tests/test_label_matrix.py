"""Tests for EmbeddingCollator label matrix construction."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from lset.data.dataset import EmbeddingDataset
from lset.data.collator import EmbeddingCollator


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("bert-base-uncased")


def _write_jsonl(samples):
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for s in samples:
        f.write(json.dumps(s) + "\n")
    f.close()
    return Path(f.name)


def test_label_matrix_shape(tokenizer):
    path = _write_jsonl([
        {"query": "a", "positives": ["b"], "negatives": ["c"]},
        {"query": "d", "positives": ["e"], "negatives": ["f"]},
    ])
    ds = EmbeddingDataset(path, tokenizer)
    collator = EmbeddingCollator(tokenizer)
    batch = collator([ds[0], ds[1]])
    assert batch["labels"].shape == (2, 4)  # 2 queries, 4 docs (1+1+1+1)


def test_positives_marked_correctly(tokenizer):
    path = _write_jsonl([
        {"query": "a", "positives": ["b"], "negatives": ["c"]},
        {"query": "d", "positives": ["e"], "negatives": ["f"]},
    ])
    ds = EmbeddingDataset(path, tokenizer)
    collator = EmbeddingCollator(tokenizer)
    batch = collator([ds[0], ds[1]])
    labels = batch["labels"]
    # Query 0's positive is doc 0, Query 1's positive is doc 2
    assert labels[0, 0] == 1.0  # q0 → doc0 positive
    assert labels[0, 1] == 0.0  # q0 → doc1 is q0's negative
    assert labels[1, 2] == 1.0  # q1 → doc2 positive
    assert labels[1, 3] == 0.0  # q1 → doc3 is q1's negative


def test_multi_positive_label_matrix(tokenizer):
    path = _write_jsonl([
        {"query": "a", "positives": ["b", "c"], "negatives": ["d"]},
    ])
    ds = EmbeddingDataset(path, tokenizer)
    collator = EmbeddingCollator(tokenizer)
    batch = collator([ds[0]])
    labels = batch["labels"]
    assert labels.shape == (1, 3)  # 1 query, 3 docs
    assert labels[0, 0] == 1.0  # first positive
    assert labels[0, 1] == 1.0  # second positive
    assert labels[0, 2] == 0.0  # negative


def test_packed_mode_with_labels(tokenizer):
    path = _write_jsonl([
        {"query": "a", "positives": ["b"], "negatives": []},
        {"query": "c", "positives": ["d"], "negatives": []},
    ])
    ds = EmbeddingDataset(path, tokenizer)
    collator = EmbeddingCollator(tokenizer, packed=True)
    batch = collator([ds[0], ds[1]])
    # Packed mode should still have labels
    assert "labels" in batch
    assert batch["labels"].shape == (2, 2)
    # Query batch should be packed
    assert "cu_seqlens" in batch["query"]
