"""Test tokenizer loading."""

from pathlib import Path

import pytest

from lset.tokenization.loader import load_tokenizer

MODEL_PATH = "/home/roy/models/Qwen3-Embedding-0.6B"


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model not available locally")
def test_load_tokenizer():
    """Tokenizer loads and encodes text."""
    tokenizer = load_tokenizer(MODEL_PATH)
    enc = tokenizer.encode("Hello world")
    assert len(enc.ids) > 0
    print(f"Tokenized 'Hello world' → {len(enc.ids)} tokens: {enc.ids}")


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model not available locally")
def test_tokenizer_roundtrip():
    """Encode then decode returns original text."""
    tokenizer = load_tokenizer(MODEL_PATH)
    text = "The quick brown fox"
    enc = tokenizer.encode(text)
    decoded = tokenizer.decode(enc.ids)
    assert decoded == text


def test_load_nonexistent():
    """Should raise FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_tokenizer("/nonexistent/path")


if __name__ == "__main__":
    test_load_tokenizer()
    test_tokenizer_roundtrip()
    print("All tokenizer tests passed!")
