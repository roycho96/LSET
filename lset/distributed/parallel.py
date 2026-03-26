"""Parallelism composition for LSET models.

Supports:
- FSDP2 only (dp_size > 1, tp_size = 1)
- TP only (tp_size > 1, dp_size = 1)
- TP + FSDP2 2D parallelism (tp_size > 1, dp_size > 1)

Application order (following TorchTitan):
1. TP: parallelize_module (if tp_size > 1)
2. Activation Checkpointing (optional)
3. FSDP2: fully_shard bottom-up
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from .mesh import build_mesh
from .tp import apply_tp


@dataclass
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    mp_dtype: torch.dtype = torch.bfloat16
    activation_checkpoint: bool = False
    ac_ratio: float = 1.0
    use_sequence_parallel: bool = False


def setup_fsdp2(model: nn.Module, dp_size: int):
    """Legacy FSDP2-only setup (backward compatible with Phase A/B).

    Args:
        model: The model to shard.
        dp_size: Data parallel world size.

    Returns:
        (model, mesh) tuple.
    """
    mesh = build_mesh(dp_size)
    _apply_fsdp2(model, mesh, torch.bfloat16)
    return model, mesh


def build_parallel_model(
    model: nn.Module,
    config,
    parallel_config: ParallelConfig,
) -> tuple[nn.Module, DeviceMesh]:
    """Apply full parallelism pipeline to a model.

    Args:
        model: nn.Module already on device with weights loaded.
        config: Model config (for TP plan).
        parallel_config: Parallelism configuration.

    Returns:
        (model, mesh) tuple.
    """
    mesh = build_mesh(
        parallel_config.dp_size,
        parallel_config.tp_size,
        parallel_config.pp_size,
    )

    # Step 1: TP (before FSDP)
    if parallel_config.tp_size > 1:
        from lset.models.decoder.qwen3.parallel_plan import get_tp_plan
        tp_mesh = mesh["tp"]
        plan = get_tp_plan(config, use_sequence_parallel=parallel_config.use_sequence_parallel)
        apply_tp(model, tp_mesh, plan)

    # Step 2: Activation Checkpointing
    if parallel_config.activation_checkpoint:
        _apply_ac(model, parallel_config.ac_ratio)

    # Step 3: FSDP2 (always apply with TP to make all params DTensors)
    if parallel_config.dp_size > 1 or parallel_config.tp_size > 1:
        dp_mesh = mesh["dp"] if mesh.ndim > 1 else mesh
        _apply_fsdp2(model, dp_mesh, parallel_config.mp_dtype)

    return model, mesh


def _apply_ac(model: nn.Module, ratio: float):
    """Selective activation checkpointing on transformer blocks."""
    from torch.utils.checkpoint import checkpoint

    if not hasattr(model, "layers"):
        return

    layers = model.layers
    n = int(len(layers) * ratio)

    for i in range(n):
        orig_forward = layers[i].forward

        def make_wrapper(fn):
            def wrapped(*args, **kwargs):
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return wrapped

        layers[i].forward = make_wrapper(orig_forward)


def _apply_fsdp2(model: nn.Module, dp_mesh: DeviceMesh, mp_dtype: torch.dtype):
    """Bottom-up FSDP2 sharding."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=mp_dtype,
        reduce_dtype=torch.float32,
    )

    if hasattr(model, "layers"):
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
