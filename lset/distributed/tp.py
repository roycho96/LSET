"""Tensor Parallelism application for LSET models."""

import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module


def apply_tp(model: nn.Module, tp_mesh: DeviceMesh, tp_plan: dict):
    """Apply Tensor Parallelism to model.

    Must be called BEFORE FSDP2.

    Args:
        model: nn.Module (on meta device or real device).
        tp_mesh: 1D DeviceMesh for the "tp" dimension.
        tp_plan: Dict mapping FQN to ParallelStyle, from get_tp_plan().
    """
    parallelize_module(model, tp_mesh, tp_plan)
