"""Tests for EmbeddingDataset multi-format support."""

import json
import tempfile

from pathlib import Path

import pytest

from tokenizers import Tokenizer

from lset.train.data.dataset import EmbeddingDataset


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("bert-base-uncased")


def _write_jsonl(samples):
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for s in samples:
        f.write(json.dumps(s) + "\n")
    f.close()
    return Path(f.name)


def test_pair_format(tokenizer):
    path = _write_jsonl(
        [
            {"query": "hello", "positive": "world"},
            {"query": "foo", "positive": "bar"},
        ]
    )
    ds = EmbeddingDataset(path, tokenizer)
    assert ds.format == "pair"
    sample = ds[0]
    assert sample["query"] == "hello"
    assert sample["positives"] == ["world"]
    assert sample["negatives"] == []
    assert sample["scores"] is None


def test_triplet_format(tokenizer):
    path = _write_jsonl(
        [
            {"query": "hello", "positive": "world", "negatives": ["bad1", "bad2"]},
        ]
    )
    ds = EmbeddingDataset(path, tokenizer)
    assert ds.format == "triplet"
    sample = ds[0]
    assert sample["positives"] == ["world"]
    assert sample["negatives"] == ["bad1", "bad2"]


def test_multi_format(tokenizer):
    path = _write_jsonl(
        [
            {"query": "hello", "positives": ["world", "earth"], "negatives": ["bad"]},
        ]
    )
    ds = EmbeddingDataset(path, tokenizer)
    assert ds.format == "multi"
    sample = ds[0]
    assert len(sample["positives"]) == 2
    assert len(sample["negatives"]) == 1


def test_scored_format(tokenizer):
    path = _write_jsonl(
        [
            {
                "query": "hello",
                "documents": [
                    {"text": "world", "score": 1.0},
                    {"text": "earth", "score": 0.5},
                    {"text": "bad", "score": 0.0},
                ],
            },
        ]
    )
    ds = EmbeddingDataset(path, tokenizer)
    assert ds.format == "scored"
    sample = ds[0]
    assert sample["scores"] == [1.0, 0.5, 0.0]
    assert len(sample["all_documents"]) == 3


def test_auto_detection(tokenizer):
    pair_path = _write_jsonl([{"query": "a", "positive": "b"}])
    triplet_path = _write_jsonl([{"query": "a", "positive": "b", "negatives": ["c"]}])
    multi_path = _write_jsonl([{"query": "a", "positives": ["b"], "negatives": ["c"]}])
    scored_path = _write_jsonl([{"query": "a", "documents": [{"text": "b", "score": 1.0}]}])

    assert EmbeddingDataset(pair_path, tokenizer).format == "pair"
    assert EmbeddingDataset(triplet_path, tokenizer).format == "triplet"
    assert EmbeddingDataset(multi_path, tokenizer).format == "multi"
    assert EmbeddingDataset(scored_path, tokenizer).format == "scored"


def test_num_hard_negatives(tokenizer):
    path = _write_jsonl(
        [
            {"query": "hello", "positive": "world", "negatives": ["a", "b", "c", "d"]},
        ]
    )
    ds = EmbeddingDataset(path, tokenizer, num_hard_negatives=2)
    sample = ds[0]
    assert len(sample["negatives"]) == 2
