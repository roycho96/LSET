"""FSDP2 setup for distributed training."""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def setup_fsdp2(model, dp_size: int):
    """Apply FSDP2 sharding to a Qwen3Decoder model.

    Args:
        model: The Qwen3Decoder model to shard.
        dp_size: Data parallel world size.

    Returns:
        (model, mesh) tuple.
    """
    mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Bottom-up: shard each layer first, then whole model
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model, mesh
