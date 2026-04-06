"""LoRA / QLoRA for embedding model fine-tuning.

LoRA: trainable low-rank adapters on frozen base weights.
QLoRA: LoRA + NF4 base weight quantization (~4x memory reduction).

References:
  Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
  Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn


# Default LoRA targets for Qwen3 (all linear projections in attn + MLP)
QWEN3_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class LoRALinear(nn.Module):
    """Linear layer with frozen base weight and trainable LoRA adapters.

    forward(x) = base_linear(x) + lora_B(lora_A(dropout(x))) * scale

    The base weight is frozen (no gradients). Only lora_A and lora_B are trained.
    Initialized so B=zeros → output starts as identity (no perturbation).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weight
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        self.r = r
        self.scale = alpha / r

        # LoRA adapters
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=base_linear.weight.dtype,
                                device=base_linear.weight.device)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=base_linear.weight.dtype,
                                device=base_linear.weight.device)

        # Init: A = kaiming uniform, B = zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def weight(self):
        """Expose base weight for compatibility with TP and other wrappers."""
        return self.base_linear.weight

    @property
    def in_features(self):
        return self.base_linear.in_features

    @property
    def out_features(self):
        return self.base_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out


def _replace_with_lora(
    model: nn.Module,
    target_set: set[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    """Replace matching nn.Linear modules with LoRALinear wrappers."""
    for fqn, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = fqn.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue

        lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)

        if "." in fqn:
            parent_fqn, child_name = fqn.rsplit(".", 1)
            parent = model.get_submodule(parent_fqn)
        else:
            parent = model
            child_name = fqn

        setattr(parent, child_name, lora_module)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] | list[str] = QWEN3_LORA_TARGETS,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace matching nn.Linear modules with LoRALinear wrappers.

    Freezes all model parameters first, then adds trainable LoRA adapters
    only on modules whose name ends with one of target_modules.

    Args:
        model: The model to modify (in-place).
        r: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Module name suffixes to target (e.g. "q_proj").
        dropout: Dropout rate on LoRA input.

    Returns:
        The modified model (same object, modified in-place).
    """
    for param in model.parameters():
        param.requires_grad_(False)

    _replace_with_lora(model, set(target_modules), r, alpha, dropout)
    return model


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
    from lset.train.quantization.nf4 import quantize_linear_nf4

    for param in model.parameters():
        param.requires_grad_(False)

    target_set = set(target_modules)

    # First pass: quantize target weights to NF4
    for fqn, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if fqn.rsplit(".", 1)[-1] not in target_set:
            continue
        quantize_linear_nf4(module, block_size=block_size, scaler_block_size=scaler_block_size)

    # Second pass: wrap with LoRA
    _replace_with_lora(model, target_set, r, alpha, dropout)

    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters (for optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights to a file."""
    lora_state = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_state[name] = param.data.clone()
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA adapter weights into a model with LoRALinear modules."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(lora_state, strict=False)
