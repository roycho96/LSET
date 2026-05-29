"""RNG state snapshot/restore for GradCache two-forward consistency."""

from __future__ import annotations

import torch

from torch.utils.checkpoint import get_device_states
from torch.utils.checkpoint import set_device_states


class RandContext:
    """Snapshot CPU + CUDA RNG at construction, restore inside ``with``."""

    def __init__(self, *tensors: torch.Tensor) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)
        self._fork = None

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None
