"""HF → LSET weight key conversion for Llama."""

from pathlib import Path

from safetensors.torch import load_file


def convert_hf_keys(hf_state_dict: dict) -> dict:
    """Strip 'model.' prefix from HF Llama weight keys."""
    converted = {}
    for key, value in hf_state_dict.items():
        new_key = key.removeprefix("model.")
        converted[new_key] = value
    return converted


def load_llama_weights(model_path: str | Path) -> dict:
    """Load Llama weights from safetensors and convert keys."""
    model_path = Path(model_path)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(str(f)))

    return convert_hf_keys(state_dict)
