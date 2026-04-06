"""Tests for PP split points and FixedLengthCollator."""

import pytest

from lset.distributed.pp import get_pp_split_points, get_stage_module_names
from lset.models.decoder.qwen3.config import Qwen3Config
from lset.train.data.collator import FixedLengthCollator


def test_pp_split_points_2_stages():
    """Split points for 2 stages with 28 layers."""
    config = Qwen3Config(num_hidden_layers=28)
    points = get_pp_split_points(config, num_stages=2)
    assert "layers.14" in points
    assert len(points) == 1


def test_pp_split_points_4_stages():
    """Split points for 4 stages with 28 layers."""
    config = Qwen3Config(num_hidden_layers=28)
    points = get_pp_split_points(config, num_stages=4)
    assert len(points) == 3
    assert "layers.7" in points
    assert "layers.14" in points
    assert "layers.21" in points


def test_pp_split_points_not_divisible():
    """Should raise if layers not divisible by stages."""
    config = Qwen3Config(num_hidden_layers=28)
    with pytest.raises(AssertionError):
        get_pp_split_points(config, num_stages=3)


def test_stage_module_names():
    """Verify module names per stage."""
    config = Qwen3Config(num_hidden_layers=4)
    stages = get_stage_module_names(config, num_stages=2)
    assert len(stages) == 2
    # First stage has embed_tokens + rotary_emb + layers 0,1
    assert "embed_tokens" in stages[0]
    assert "rotary_emb" in stages[0]
    assert "layers.0" in stages[0]
    assert "layers.1" in stages[0]
    # Last stage has layers 2,3 + norm
    assert "layers.2" in stages[1]
    assert "layers.3" in stages[1]
    assert "norm" in stages[1]


def test_fixed_length_collator_shape():
    """FixedLengthCollator should always produce same-shaped tensors."""
    collator = FixedLengthCollator(pad_token_id=0, max_seq_length=16)

    batch = [
        {"query": {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
         "positive": {"input_ids": [4, 5], "attention_mask": [1, 1]}},
        {"query": {"input_ids": [6, 7, 8, 9, 10], "attention_mask": [1, 1, 1, 1, 1]},
         "positive": {"input_ids": [11], "attention_mask": [1]}},
    ]
    result = collator(batch)
    assert result["query"]["input_ids"].shape == (2, 16)
    assert result["doc"]["input_ids"].shape == (2, 16)
    assert result["query"]["attention_mask"].shape == (2, 16)


def test_fixed_length_truncation():
    """FixedLengthCollator should truncate to max_seq_length."""
    collator = FixedLengthCollator(pad_token_id=0, max_seq_length=4)
    batch = [
        {"query": {"input_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1, 1, 1, 1, 1, 1]},
         "positive": {"input_ids": [7, 8], "attention_mask": [1, 1]}},
    ]
    result = collator(batch)
    assert result["query"]["input_ids"].shape == (1, 4)
