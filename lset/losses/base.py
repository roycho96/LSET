"""Shared primitives for contrastive losses.

``LogitScale`` holds a learnable, clamped inverse-temperature (CLIP style).
``log(scale)`` is the optimized parameter; its ``.exp().clamp(max=max_scale)``
is multiplied into the cosine similarity matrix. Clamping prevents a runaway
scale from overflowing bf16/fp16 at the softmax boundary (>~11.0 in log space
already saturates bf16 exp to inf).

Use as a module attribute on the task that owns the loss so it gets
registered in ``state_dict`` and the optimizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogitScale(nn.Module):
    """Learnable CLIP-style logit scale with max-clamp.

    Args:
        init_scale: initial value of ``exp(log_scale)`` (i.e. ``1 / temperature``).
        max_scale:  hard clamp on the returned scale to avoid overflow.
        learnable:  if False, ``log_scale`` is a buffer and not optimized.
    """

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
