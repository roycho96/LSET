"""MXFP8 (Microscaling FP8) training support — experimental.

Uses torchao's prototype MX format for Blackwell-native MXFP8 training.
This is a prototype feature and may be unstable.

Key constraints:
- Requires SM100 (Blackwell) for native acceleration; emulated elsewhere
- Prototype API — subject to change between torchao versions
- torch.compile recommended for performance
- Block size is fixed at 32 (MX spec)
"""

from __future__ import annotations

import torch.nn as nn


# Valid MXFP8 recipes
VALID_MX_RECIPES = ("mxfp8_emulated", "mxfp8_cublas", "mxfp8_cublas_rceil")


def apply_mxfp8_training(
    model: nn.Module,
    recipe: str = "mxfp8_emulated",
) -> nn.Module:
    """Convert model's linear layers to MXLinear for MXFP8 training.

    Uses torchao's prototype MX format. Weights stay in high precision;
    quantization happens dynamically during forward/backward.

    Args:
        model: Model to convert in-place.
        recipe: One of "mxfp8_emulated", "mxfp8_cublas", "mxfp8_cublas_rceil".

    Returns:
        The converted model.
    """
    if recipe not in VALID_MX_RECIPES:
        raise ValueError(f"Invalid MXFP8 recipe '{recipe}'. Choose from: {VALID_MX_RECIPES}")

    from torchao.prototype.mx_formats.mx_linear import MXLinearConfig
    from torchao.quantization.quant_api import quantize_

    config = MXLinearConfig.from_recipe_name(recipe)
    quantize_(model, config)

    return model
