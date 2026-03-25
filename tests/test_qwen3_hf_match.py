"""Compare LSET Qwen3 output with HuggingFace transformers."""

import pytest
import torch

MODEL_PATH = "/home/roy/models/Qwen3-Embedding-0.6B"


def _hf_available():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _hf_available(), reason="transformers not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hf_match():
    """Load same weights in both LSET and HF, compare hidden_states."""
    from transformers import AutoModel, AutoTokenizer
    from lset.models.decoder.qwen3.config import Qwen3Config
    from lset.models.decoder.qwen3.model import Qwen3Decoder
    from lset.models.decoder.qwen3.weights import load_qwen3_weights

    device = torch.device("cuda:0")
    dtype = torch.float32  # use fp32 for comparison precision

    # LSET model
    config = Qwen3Config.from_hf_json(f"{MODEL_PATH}/config.json")
    lset_model = Qwen3Decoder(config).to(device=device, dtype=dtype)
    state_dict = load_qwen3_weights(MODEL_PATH)
    lset_model.load_state_dict(state_dict, strict=True)
    lset_model.eval()

    # HF model
    hf_model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    hf_model.eval()

    # Tokenize
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    text = "Hello, this is a test sentence for embedding comparison."
    inputs = hf_tokenizer(text, return_tensors="pt", padding=False).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Forward
    with torch.no_grad():
        lset_out = lset_model(input_ids, attention_mask)
        hf_out = hf_model(input_ids, attention_mask=attention_mask)

    lset_hidden = lset_out["hidden_states"]
    hf_hidden = hf_out.last_hidden_state

    assert lset_hidden.shape == hf_hidden.shape, (
        f"Shape mismatch: LSET {lset_hidden.shape} vs HF {hf_hidden.shape}"
    )

    # Compare
    max_diff = (lset_hidden - hf_hidden).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6e}")
    assert torch.allclose(lset_hidden, hf_hidden, atol=1e-4), (
        f"Hidden states differ. Max diff: {max_diff:.6e}"
    )
    print("HF match test passed!")


if __name__ == "__main__":
    test_hf_match()
