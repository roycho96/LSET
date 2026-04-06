"""Float8 (FP8) mixed-precision training using torchao.

Converts nn.Linear modules to Float8Linear for ~1.5x training speedup
on Hopper+ (SM≥8.9) and Blackwell (SM100) GPUs.

Key constraints:
- Requires torch.compile for performance benefit
- All linear dims must be divisible by 16
- Not compatible with LoRA (torchtune#2833)
- Rowwise recipe recommended for best accuracy
"""

from __future__ import annotations

import torch.nn as nn

# Valid recipe names
VALID_RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")


def apply_fp8_training(
    model: nn.Module,
    recipe: str = "rowwise",
    enable_fsdp_float8_all_gather: bool = False,
) -> nn.Module:
    """Convert model's linear layers to Float8Linear for FP8 training.

    Must be called BEFORE TP and FSDP2 (composition order:
    FP8 → TP → AC → FSDP2 → compile → weights).

    Args:
        model: Model to convert in-place.
        recipe: One of "tensorwise", "rowwise", "rowwise_with_gw_hp".
        enable_fsdp_float8_all_gather: Enable FP8 all-gather in FSDP2
            (only for tensorwise recipe).

    Returns:
        The converted model.
    """
    if recipe not in VALID_RECIPES:
        raise ValueError(f"Invalid FP8 recipe '{recipe}'. Choose from: {VALID_RECIPES}")

    from torchao.float8.config import Float8LinearConfig
    from torchao.float8.float8_linear_utils import convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name(recipe)

    if enable_fsdp_float8_all_gather and recipe == "tensorwise":
        # Override to enable FP8 all-gather for tensorwise recipe
        from dataclasses import replace

        config = replace(config, enable_fsdp_float8_all_gather=True)

    # For rowwise recipe, enable precision cast emulation (pytorch#150859)
    if recipe in ("rowwise", "rowwise_with_gw_hp"):
        import torch._inductor.config

        torch._inductor.config.emulate_precision_casts = True

    # Filter: skip modules with dims not divisible by 16
    def _filter_fn(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        return True

    convert_to_float8_training(model, config=config, module_filter_fn=_filter_fn)

    return model


def get_fp8_tp_plan(config, use_fp8: bool = False, recipe: str = "rowwise"):
    """Get TP plan that uses Float8-aware parallel styles for tensorwise recipe.

    For rowwise recipe, uses standard ColwiseParallel/RowwiseParallel since
    Float8 all-gather only works with tensorwise scaling.
    """
    from lset.models.decoder.qwen3.parallel_plan import get_tp_plan

    if not use_fp8 or recipe != "tensorwise":
        return get_tp_plan(config)

    # Tensorwise FP8 uses Float8ColwiseParallel/Float8RowwiseParallel
    from torchao.float8.float8_tensor_parallel import Float8ColwiseParallel
    from torchao.float8.float8_tensor_parallel import Float8RowwiseParallel

    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update(
            {
                f"{p}.self_attn.q_proj": Float8ColwiseParallel(),
                f"{p}.self_attn.k_proj": Float8ColwiseParallel(),
                f"{p}.self_attn.v_proj": Float8ColwiseParallel(),
                f"{p}.self_attn.o_proj": Float8RowwiseParallel(),
                f"{p}.mlp.gate_proj": Float8ColwiseParallel(),
                f"{p}.mlp.up_proj": Float8ColwiseParallel(),
                f"{p}.mlp.down_proj": Float8RowwiseParallel(),
            }
        )
    return plan
