"""Shared primitives for contrastive losses."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogitScale(nn.Module):
    """Learnable CLIP-style logit scale with max-clamp."""

    def __init__(self, init_scale: float = 20.0, max_scale: float = 100.0, learnable: bool = True):
        super().__init__()
        if init_scale <= 0:
            raise ValueError(f"init_scale must be > 0, got {init_scale}")
        log_init = torch.tensor(math.log(init_scale), dtype=torch.float32)
        if learnable:
            self.log_scale = nn.Parameter(log_init)
        else:
            self.register_buffer("log_scale", log_init)
        self.max_scale = float(max_scale)

    @classmethod
    def from_temperature(cls, temperature: float, **kw) -> "LogitScale":
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        return cls(init_scale=1.0 / temperature, **kw)

    def forward(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=self.max_scale)

    def extra_repr(self) -> str:
        s = float(self.log_scale.detach().exp().clamp(max=self.max_scale).item())
        return f"scale={s:.4f}, max_scale={self.max_scale}, learnable={isinstance(self.log_scale, nn.Parameter)}"
