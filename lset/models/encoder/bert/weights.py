"""Weight loading for BERT and XLM-RoBERTa models."""

from pathlib import Path

import torch
from safetensors.torch import load_file


def _load_weights(model_path: Path) -> dict:
    """Load weights from safetensors or pytorch_model.bin."""
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(str(f)))
        return state_dict

    bin_files = sorted(model_path.glob("pytorch_model*.bin"))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            state_dict.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return state_dict

    raise FileNotFoundError(f"No model files found in {model_path}")


def load_bert_weights(model_path: str | Path) -> dict:
    """Load BERT weights and convert keys to LSET format.

    BERT HF keys use 'bert.' prefix and 'gamma'/'beta' for LayerNorm.
    Maps to our structure: embeddings.*, layers.{i}.attention.*, layers.{i}.mlp.*
    """
    model_path = Path(model_path)
    hf = _load_weights(model_path)

    converted = {}
    for key, value in hf.items():
        new_key = key
        # Strip 'bert.' prefix
        new_key = new_key.removeprefix("bert.")
        # Skip pooler (not used for embeddings)
        if new_key.startswith("pooler."):
            continue
        # Skip cls head (for masked LM)
        if new_key.startswith("cls."):
            continue
        # LayerNorm gamma/beta → weight/bias
        new_key = new_key.replace(".gamma", ".weight").replace(".beta", ".bias")
        # encoder.layer.{i} → layers.{i}
        new_key = new_key.replace("encoder.layer.", "layers.")
        # attention.self.query → attention.query
        new_key = new_key.replace("attention.self.", "attention.")
        # attention.output.dense → attention.dense
        new_key = new_key.replace("attention.output.dense", "attention.dense")
        # attention.output.LayerNorm → attention.LayerNorm
        new_key = new_key.replace("attention.output.LayerNorm", "attention.LayerNorm")
        # intermediate.dense → mlp.dense_in
        new_key = new_key.replace("intermediate.dense", "mlp.dense_in")
        # output.dense → mlp.dense_out (but not attention.output.dense)
        new_key = new_key.replace("output.dense", "mlp.dense_out")
        # output.LayerNorm → mlp.LayerNorm
        new_key = new_key.replace("output.LayerNorm", "mlp.LayerNorm")

        converted[new_key] = value
    return converted


def load_xlm_roberta_weights(model_path: str | Path) -> dict:
    """Load XLM-RoBERTa weights and convert keys to LSET format.

    XLM-RoBERTa uses same structure as BERT but without 'bert.' prefix,
    and standard weight/bias naming for LayerNorm.
    """
    model_path = Path(model_path)
    hf = _load_weights(model_path)

    converted = {}
    for key, value in hf.items():
        new_key = key
        # Skip pooler and classifier heads
        if any(new_key.startswith(p) for p in ("pooler.", "classifier.", "lm_head.")):
            continue
        # encoder.layer.{i} → layers.{i}
        new_key = new_key.replace("encoder.layer.", "layers.")
        # attention.self.query → attention.query
        new_key = new_key.replace("attention.self.", "attention.")
        # attention.output.dense → attention.dense
        new_key = new_key.replace("attention.output.dense", "attention.dense")
        new_key = new_key.replace("attention.output.LayerNorm", "attention.LayerNorm")
        # intermediate.dense → mlp.dense_in
        new_key = new_key.replace("intermediate.dense", "mlp.dense_in")
        new_key = new_key.replace("output.dense", "mlp.dense_out")
        new_key = new_key.replace("output.LayerNorm", "mlp.LayerNorm")

        converted[new_key] = value
    return converted
