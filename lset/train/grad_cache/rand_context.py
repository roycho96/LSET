"""RNG state snapshot/restore for GradCache two-forward consistency.

GradCache runs the same minibatch through the encoder twice:
  - Step 1 (no_grad): cache embeddings
  - Step 3 (with_grad): replay forward and backward cached grads into params

Dropout / any stochastic op must produce identical outputs in both passes,
otherwise the cached gradient is computed against a different function than
the one being replayed, which silently biases the training signal.

RandContext snapshots CPU + CUDA RNG at Step 1 and restores it on Step 3.
`fork_rng` sandboxes the restored state so the surrounding training loop's
RNG sequence is not rewound.
"""

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
