"""Parallelism composition for LSET models.

Supports:
- FSDP2 only (dp_size > 1, tp_size = 1)
- TP only (tp_size > 1, dp_size = 1)
- TP + FSDP2 2D parallelism (tp_size > 1, dp_size > 1)

Application order (following TorchTitan):
1. TP: parallelize_module (if tp_size > 1)
2. Activation Checkpointing (optional, selective)
3. FSDP2: fully_shard bottom-up + forward/backward prefetch
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import fully_shard

from lset.distributed.mesh import build_mesh
from lset.distributed.tp import apply_tp


@dataclass
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    mp_dtype: torch.dtype = torch.bfloat16
    activation_checkpoint: bool = False
    ac_ratio: float = 1.0
    ac_mode: str = "selective"  # "selective" (op-level) or "full" (whole layer)
    use_sequence_parallel: bool = False
    use_lora: bool = False
    async_tp: bool = False  # no-op unless torch.compile drives the pipeline pass


def setup_fsdp2(model: nn.Module, dp_size: int):
    """Legacy FSDP2-only setup (backward compatible)."""
    mesh = build_mesh(dp_size)
    _apply_fsdp2(model, mesh, torch.bfloat16)
    return model, mesh


def build_parallel_model(
    model: nn.Module,
    config,
    parallel_config: ParallelConfig,
) -> tuple[nn.Module, DeviceMesh]:
    """Apply full parallelism pipeline to a model."""
    mesh = build_mesh(
        parallel_config.dp_size,
        parallel_config.tp_size,
        parallel_config.pp_size,
    )

    # Step 1: TP (before FSDP).
    if parallel_config.tp_size > 1:
        from lset.models.decoder.qwen3.parallel_plan import get_tp_plan

        tp_mesh = mesh["tp"]
        plan = get_tp_plan(
            config,
            use_sequence_parallel=parallel_config.use_sequence_parallel,
            use_lora=parallel_config.use_lora,
        )
        apply_tp(model, tp_mesh, plan)

        # Async-TP fusion happens in the inductor pass; no-op without compile.
        if parallel_config.async_tp:
            import torch._inductor.config as ic

            ic._micro_pipeline_tp = True

    # Step 2: Activation Checkpointing (selective or full).
    if parallel_config.activation_checkpoint:
        _apply_ac(model, parallel_config.ac_ratio, parallel_config.ac_mode)

    # Step 3: FSDP2 (always apply with TP to make all params DTensors).
    if parallel_config.dp_size > 1 or parallel_config.tp_size > 1:
        dp_mesh = mesh["dp"] if mesh.ndim > 1 else mesh
        _apply_fsdp2(model, dp_mesh, parallel_config.mp_dtype)

    return model, mesh


def _apply_ac(model: nn.Module, ratio: float = 1.0, mode: str = "selective") -> None:
    """Wrap transformer blocks with selective (op-level SAC) or full AC."""
    if not hasattr(model, "layers"):
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    layers = model.layers
    n = max(1, int(len(layers) * ratio))

    if mode == "selective":
        # Save matmul/SDPA outputs, recompute the rest. Triton kernels register
        # as custom (non-aten) ops, so they fall through to PREFER_RECOMPUTE.
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        save_ops = {
            torch.ops.aten.mm.default: CheckpointPolicy.MUST_SAVE,
            torch.ops.aten.addmm.default: CheckpointPolicy.MUST_SAVE,
            torch.ops.aten.bmm.default: CheckpointPolicy.MUST_SAVE,
            torch.ops.aten._scaled_dot_product_flash_attention.default:
                CheckpointPolicy.MUST_SAVE,
            torch.ops.aten._scaled_dot_product_efficient_attention.default:
                CheckpointPolicy.MUST_SAVE,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default:
                CheckpointPolicy.MUST_SAVE,
        }

        def _policy(ctx, func, *args, **kwargs):
            return save_ops.get(func, CheckpointPolicy.PREFER_RECOMPUTE)

        def _ctx():
            return create_selective_checkpoint_contexts(_policy)

        for i in range(n):
            layers[i] = ptd_checkpoint_wrapper(
                layers[i], context_fn=_ctx, preserve_rng_state=True
            )
    else:
        # Full AC — simpler, higher memory savings, higher recompute cost.
        for i in range(n):
            layers[i] = ptd_checkpoint_wrapper(layers[i], preserve_rng_state=True)


def _apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_dtype: torch.dtype,
    enable_prefetch: bool = True,
) -> None:
    """Bottom-up FSDP2 sharding with forward + backward prefetch."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=mp_dtype,
        reduce_dtype=torch.float32,
    )

    layers = list(getattr(model, "layers", []))
    if layers:
        for i, layer in enumerate(layers):
            # Last layer: keep its params gathered for the imminent backward.
            reshard = i < len(layers) - 1
            fully_shard(
                layer, mesh=dp_mesh, mp_policy=mp_policy,
                reshard_after_forward=reshard,
            )
    # Root. Reshard root weights (embed_tokens, final norm) after forward.
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)

    if enable_prefetch and len(layers) > 1:
        # Forward: prefetch layer N+1 while computing layer N.
        for i in range(len(layers) - 1):
            layers[i].set_modules_to_forward_prefetch([layers[i + 1]])
        # Backward: prefetch layer N-1 while computing layer N's grads.
        rev = list(reversed(layers))
        for i in range(len(rev) - 1):
            rev[i].set_modules_to_backward_prefetch([rev[i + 1]])
