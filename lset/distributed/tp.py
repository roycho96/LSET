"""Tensor Parallelism application for LSET models."""

import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module


def apply_tp(model: nn.Module, tp_mesh: DeviceMesh, tp_plan: dict):
    """Apply Tensor Parallelism to model."""
    parallelize_module(model, tp_mesh, tp_plan)
