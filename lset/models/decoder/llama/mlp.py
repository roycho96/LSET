"""Llama MLP — identical to Qwen3 SwiGLU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
