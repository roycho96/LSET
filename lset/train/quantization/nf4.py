"""NF4 (4-bit NormalFloat) weight quantization for QLoRA.

Wraps torchao's NF4 tensor conversion with automatic scaler_block_size
computation for double quantization compatibility.

Reference: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
"""

from __future__ import annotations

import torch.nn as nn


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


def quantize_linear_nf4(
    module: nn.Linear,
    block_size: int = 64,
    scaler_block_size: int = 256,
) -> None:
    """Quantize a Linear module's weight to NF4 in-place.

    After this call, module.weight is an NF4Tensor (requires_grad=False).

    Args:
        module: Linear layer to quantize.
        block_size: NF4 quantization block size.
        scaler_block_size: NF4 double-quantization scaler block size.
    """
    from torchao.dtypes.nf4tensor import to_nf4

    weight_data = module.weight.data
    sbs = _compute_scaler_block_size(weight_data.numel(), block_size, scaler_block_size)
    nf4_weight = to_nf4(weight_data, block_size=block_size, scaler_block_size=sbs)
    module.weight = nn.Parameter(nf4_weight, requires_grad=False)
