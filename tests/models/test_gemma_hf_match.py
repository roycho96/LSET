"""Compare EmbeddingGemma model output with HuggingFace transformers."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma_hf_match():
    """Verify LSET EmbeddingGemma matches HF output."""
    model_path = "/home/roy/models/embeddinggemma-300m"

    # --- HF model ---
    from transformers import AutoModel
    from transformers import AutoTokenizer

    hf_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_hidden = hf_out.last_hidden_state

    # --- LSET model ---
    from lset.models import get_model_spec

    spec = get_model_spec("embeddinggemma")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    lset_model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    lset_model.load_state_dict(state_dict, strict=True)
    lset_model = lset_model.cuda().eval()

    with torch.no_grad():
        lset_out = lset_model(inputs["input_ids"], inputs["attention_mask"])
    lset_hidden = lset_out["hidden_states"]

    diff = (hf_hidden - lset_hidden).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Gemma HF match: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"HF shape: {hf_hidden.shape}, LSET shape: {lset_hidden.shape}")

    # Float32 model — should be very close
    assert max_diff < 0.01, f"Max diff too large: {max_diff}"
