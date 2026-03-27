"""QLoRA: Quantized LoRA using torchao NF4 for 4-bit base weights.

Quantizes all target linear weights to NF4 (~4x memory reduction),
then applies LoRA adapters on top. Only LoRA params are trainable.

Reference: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
"""

from __future__ import annotations

import torch.nn as nn

from lset.modules.lora import LoRALinear, QWEN3_LORA_TARGETS


def _compute_scaler_block_size(numel: int, block_size: int, default_scaler_block_size: int = 256) -> int:
    """Compute a valid scaler_block_size that divides the number of scalers.

    NF4 double quantization requires: (numel / block_size) % scaler_block_size == 0
    """
    n_scalers = numel // block_size
    if n_scalers % default_scaler_block_size == 0:
        return default_scaler_block_size
    # Find largest power-of-2 divisor ≤ default
    sbs = default_scaler_block_size
    while sbs > 1 and n_scalers % sbs != 0:
        sbs //= 2
    return max(sbs, 1)


def apply_qlora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] | list[str] = QWEN3_LORA_TARGETS,
    dropout: float = 0.0,
    block_size: int = 64,
    scaler_block_size: int = 256,
) -> nn.Module:
    """Apply QLoRA: NF4 quantize all target linears, then add LoRA adapters.

    Steps:
    1. Freeze all parameters
    2. For each target linear: quantize weight to NF4
    3. Wrap with LoRALinear (trainable low-rank adapters)

    Args:
        model: Model to modify in-place.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Module name suffixes to target.
        dropout: Dropout rate on LoRA input.
        block_size: NF4 quantization block size.
        scaler_block_size: NF4 double-quantization scaler block size.

    Returns:
        The modified model.
    """
    from torchao.dtypes.nf4tensor import to_nf4

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    target_set = set(target_modules)

    for fqn, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = fqn.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue

        # Quantize weight to NF4
        weight_data = module.weight.data
        sbs = _compute_scaler_block_size(weight_data.numel(), block_size, scaler_block_size)
        nf4_weight = to_nf4(weight_data, block_size=block_size, scaler_block_size=sbs)
        module.weight = nn.Parameter(nf4_weight, requires_grad=False)

        # Wrap with LoRA
        lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)

        # Replace in parent
        if "." in fqn:
            parent_fqn, child_name = fqn.rsplit(".", 1)
            parent = model.get_submodule(parent_fqn)
        else:
            parent = model
            child_name = fqn

        setattr(parent, child_name, lora_module)

    return model
