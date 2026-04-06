"""Tests for model registry and auto-detection."""

from lset.models.registry import ModelSpec
from lset.models.registry import detect_model_type
from lset.models.registry import get_model_spec


def test_all_models_registered():
    """All 5 model types should be in the registry."""
    for name in ["qwen3", "llama", "bert", "xlm-roberta", "embeddinggemma"]:
        spec = get_model_spec(name)
        assert isinstance(spec, ModelSpec)


def test_aliases():
    """Aliases should resolve correctly."""
    assert get_model_spec("qwen3-embedding") is get_model_spec("qwen3")
    assert get_model_spec("llama-nemotron-embed") is get_model_spec("llama")
    assert get_model_spec("nv-embed") is get_model_spec("llama")
    assert get_model_spec("bge-m3") is get_model_spec("xlm-roberta")


def test_auto_detect_qwen3():
    assert detect_model_type("/home/roy/models/Qwen3-Embedding-0.6B") == "qwen3"


def test_auto_detect_llama():
    assert detect_model_type("/home/roy/models/llama-nemotron-embed-1b-v2") == "llama"


def test_auto_detect_bert():
    assert detect_model_type("/home/roy/models/bert-base-uncased") == "bert"


def test_auto_detect_bge_m3():
    assert detect_model_type("/home/roy/models/bge-m3") == "xlm-roberta"


def test_auto_detect_gemma():
    assert detect_model_type("/home/roy/models/embeddinggemma-300m") == "embeddinggemma"


def test_pooling_defaults():
    assert get_model_spec("qwen3").default_pooling == "last_token"
    assert get_model_spec("llama").default_pooling == "mean"
    assert get_model_spec("bert").default_pooling == "cls"
    assert get_model_spec("xlm-roberta").default_pooling == "cls"
    assert get_model_spec("embeddinggemma").default_pooling == "mean"


def test_padding_defaults():
    assert get_model_spec("qwen3").default_padding_side == "left"
    assert get_model_spec("llama").default_padding_side == "right"
    assert get_model_spec("bert").default_padding_side == "right"
    assert get_model_spec("xlm-roberta").default_padding_side == "right"
    assert get_model_spec("embeddinggemma").default_padding_side == "right"
