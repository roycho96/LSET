"""Weight loading for EmbeddingGemma."""

from pathlib import Path

import torch

from safetensors.torch import load_file


def load_gemma_weights(model_path: str | Path) -> dict:
    """Load EmbeddingGemma weights and convert keys.

    Keys in safetensors have no prefix — directly match our module names.
    """
    model_path = Path(model_path)
    safetensor_files = sorted(model_path.glob("*.safetensors"))

    # Main model weights
    state_dict = {}
    for f in safetensor_files:
        if f.name.startswith("model") or f.name == "model.safetensors":
            state_dict.update(load_file(str(f)))

    # No prefix stripping needed — keys match directly
    converted = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.")
        converted[new_key] = value

    return converted


def load_gemma_projection(model_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the 2_Dense and 3_Dense post-pooling projection weights.

    Returns (proj_up_weight, proj_down_weight) for 768→3072→768 projection.
    """
    model_path = Path(model_path)
    proj_up = load_file(str(model_path / "2_Dense" / "model.safetensors"))
    proj_down = load_file(str(model_path / "3_Dense" / "model.safetensors"))
    # Sentence-transformers saves as "linear.weight"
    return proj_up["linear.weight"], proj_down["linear.weight"]
