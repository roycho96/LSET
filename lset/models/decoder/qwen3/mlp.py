from __future__ import annotations

import torch
import torch.nn as nn

from lset.kernels import swiglu
from lset.kernels.fused_gate_up_swiglu import FusedGateUpSwiGLU
from lset.models.decoder.qwen3.config import Qwen3Config


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, fused_gate_up: bool = False):
        super().__init__()
        self.fused_gate_up = fused_gate_up
        if fused_gate_up:
            self.gate_up_proj = FusedGateUpSwiGLU(
                config.hidden_size, config.intermediate_size, bias=False
            )
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up:
            # gate_up_proj already returns silu(gate) * up.
            return self.down_proj(self.gate_up_proj(x))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
