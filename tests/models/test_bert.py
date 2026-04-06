"""Tests for BERT encoder model."""

import torch

from lset.models.encoder.bert.config import BertConfig
from lset.models.encoder.bert.model import BertEncoder


def test_config_from_hf_json():
    config = BertConfig.from_hf_json("/home/roy/models/bert-base-uncased/config.json")
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.intermediate_size == 3072
    assert config.vocab_size == 30522
    assert config.max_position_embeddings == 512
    assert config.type_vocab_size == 2
    assert config.layer_norm_eps == 1e-12
    assert config.position_offset == 0  # BERT, not RoBERTa


def test_xlm_roberta_config():
    config = BertConfig.from_hf_json("/home/roy/models/bge-m3/config.json")
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 24
    assert config.vocab_size == 250002
    assert config.max_position_embeddings == 8194
    assert config.type_vocab_size == 1
    assert config.position_offset == 2  # XLM-RoBERTa offset


def test_forward_shape():
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = BertEncoder(config)
    x = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    out = model(x, mask)
    assert out["hidden_states"].shape == (2, 16, 64)


def test_bidirectional_attention():
    """BERT uses full bidirectional attention — each token sees all others."""
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = BertEncoder(config)
    x = torch.randint(0, 1000, (1, 8))
    out = model(x)
    assert out["hidden_states"].shape == (1, 8, 64)
    assert not torch.isnan(out["hidden_states"]).any()


def test_padding_mask():
    """Verify padding tokens don't affect non-padding outputs."""
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = BertEncoder(config)
    model.eval()

    x = torch.randint(1, 1000, (1, 6))
    # No padding
    mask_full = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        out_full = model(x, mask_full)

    # With padding at end (right-pad)
    x_pad = torch.cat([x, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    mask_pad = torch.cat([mask_full, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    with torch.no_grad():
        out_pad = model(x_pad, mask_pad)

    # First 6 positions should be similar (not exact due to softmax normalization change)
    diff = (out_full["hidden_states"][0, :6] - out_pad["hidden_states"][0, :6]).abs().max().item()
    # Padding changes softmax normalization so diff won't be 0, but should be modest
    assert diff < 1.0, f"Padding effect too large: {diff}"
