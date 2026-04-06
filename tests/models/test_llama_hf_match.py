"""Compare Llama model output with HuggingFace transformers."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hf_match():
    """Verify LSET Llama matches HF output within tolerance."""
    model_path = "/home/roy/models/llama-nemotron-embed-1b-v2"

    # --- HF model ---
    from transformers import AutoModel, AutoTokenizer
    hf_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    hf_model = hf_model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_hidden = hf_out.last_hidden_state  # [1, S, H]

    # --- LSET model ---
    from lset.models import get_model_spec
    spec = get_model_spec("llama")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    lset_model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    lset_model.load_state_dict(state_dict, strict=True)
    lset_model = lset_model.to(device="cuda", dtype=torch.bfloat16).eval()

    with torch.no_grad():
        lset_out = lset_model(inputs["input_ids"], inputs["attention_mask"])
    lset_hidden = lset_out["hidden_states"]

    # Compare
    diff = (hf_hidden - lset_hidden).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Llama HF match: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # bfloat16 tolerance — Llama3 RoPE scaling has minor numerical
    # differences in inv_freq computation vs HF. Diff accumulates over
    # 16 layers but stays small (mean < 0.05).
    assert max_diff < 0.5, f"Max diff too large: {max_diff}"
    assert mean_diff < 0.05, f"Mean diff too large: {mean_diff}"

    del hf_model
    torch.cuda.empty_cache()
