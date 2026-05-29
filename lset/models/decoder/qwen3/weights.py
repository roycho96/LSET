"""HF -> LSET weight key conversion for Qwen3."""

from pathlib import Path

import torch

from safetensors.torch import load_file


def convert_hf_keys(hf_state_dict: dict) -> dict:
    """Strip 'model.' prefix from HF Qwen3 weight keys."""
    converted = {}
    for key, value in hf_state_dict.items():
        new_key = key.removeprefix("model.")
        converted[new_key] = value
    return converted


def _fuse_qkv_weights(state_dict: dict) -> dict:
    """Fuse separate Q/K/V weights into qkv_proj and gate/up into gate_up_proj."""
    # First pass: identify keys to fuse
    skip_keys = set()
    fuse_ops = []  # (fused_key, [source_keys_in_order])

    for key in state_dict:
        if ".q_proj.weight" in key:
            prefix = key.replace("q_proj.weight", "")
            k_key = f"{prefix}k_proj.weight"
            v_key = f"{prefix}v_proj.weight"
            if k_key in state_dict and v_key in state_dict:
                fuse_ops.append((f"{prefix}qkv_proj.weight", [key, k_key, v_key]))
                skip_keys.update([key, k_key, v_key])

        if ".gate_proj.weight" in key:
            prefix = key.replace("gate_proj.weight", "")
            up_key = f"{prefix}up_proj.weight"
            if up_key in state_dict:
                fuse_ops.append((f"{prefix}gate_up_proj.weight", [key, up_key]))
                skip_keys.update([key, up_key])

    # Second pass: build fused dict
    fused = {}
    for key, value in state_dict.items():
        if key not in skip_keys:
            fused[key] = value

    for fused_key, source_keys in fuse_ops:
        fused[fused_key] = torch.cat([state_dict[k] for k in source_keys], dim=0)

    return fused


def load_qwen3_weights(model_path: str | Path, fused_projections: bool = False) -> dict:
    """Load Qwen3 weights from safetensors and convert keys."""
    model_path = Path(model_path)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(str(f)))

    converted = convert_hf_keys(state_dict)

    if fused_projections:
        converted = _fuse_qkv_weights(converted)

    return converted
