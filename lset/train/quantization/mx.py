"""MXFP8 (Microscaling FP8) training support — experimental."""

from __future__ import annotations

import torch.nn as nn

# Valid MXFP8 recipes
VALID_MX_RECIPES = ("mxfp8_emulated", "mxfp8_cublas", "mxfp8_cublas_rceil")


def apply_mxfp8_training(
    model: nn.Module,
    recipe: str = "mxfp8_emulated",
) -> nn.Module:
    """Convert model's linear layers to MXLinear for MXFP8 training."""
    if recipe not in VALID_MX_RECIPES:
        raise ValueError(f"Invalid MXFP8 recipe '{recipe}'. Choose from: {VALID_MX_RECIPES}")

    from torchao.prototype.mx_formats.mx_linear import MXLinearConfig
    from torchao.quantization.quant_api import quantize_

    config = MXLinearConfig.from_recipe_name(recipe)
    quantize_(model, config)

    return model
