"""Compare BERT model output with HuggingFace transformers."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bert_hf_match():
    """Verify LSET BERT matches HF output."""
    model_path = "/home/roy/models/bert-base-uncased"

    # --- HF model ---
    from transformers import BertModel
    from transformers import BertTokenizer

    hf_model = BertModel.from_pretrained(model_path).cuda().eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)

    text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_hidden = hf_out.last_hidden_state

    # --- LSET model ---
    from lset.models import get_model_spec

    spec = get_model_spec("bert")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    lset_model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    lset_model.load_state_dict(state_dict, strict=True)
    lset_model = lset_model.cuda().eval()

    with torch.no_grad():
        lset_out = lset_model(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )
    lset_hidden = lset_out["hidden_states"]

    diff = (hf_hidden - lset_hidden).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"BERT HF match: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    assert max_diff < 1e-4, f"Max diff too large: {max_diff}"
    torch.testing.assert_close(hf_hidden, lset_hidden, atol=1e-4, rtol=1e-4)

    del hf_model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bge_m3_hf_match():
    """Verify LSET XLM-RoBERTa (BGE-M3) matches HF output."""
    model_path = "/home/roy/models/bge-m3"

    # --- HF model ---
    from transformers import AutoModel
    from transformers import AutoTokenizer

    hf_model = AutoModel.from_pretrained(model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_hidden = hf_out.last_hidden_state

    # --- LSET model ---
    from lset.models import get_model_spec

    spec = get_model_spec("xlm-roberta")
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
    print(f"BGE-M3 HF match: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    assert max_diff < 1e-4, f"Max diff too large: {max_diff}"

    del hf_model
    torch.cuda.empty_cache()
