"""Tensor Parallel plan for Qwen3 model.

Following TorchTitan/TorchTune pattern:
- ColwiseParallel for Q/K/V projections (shard output dim = heads)
- RowwiseParallel for O projection (shard input dim = heads)
- ColwiseParallel for gate/up projections
- RowwiseParallel for down projection

Note: We skip SequenceParallel on norms for simplicity. This means an extra
all-reduce at RowwiseParallel outputs, but avoids complexity with RoPE and
custom attention patterns (packed mode, flash_attn_varlen_func).
"""

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)


def get_tp_plan(config) -> dict:
    """Get TP parallelization plan for Qwen3Decoder.

    Args:
        config: Qwen3Config with num_hidden_layers.

    Returns:
        Dict mapping FQN patterns to ParallelStyle instances.
    """
    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update({
            # Q/K/V: Colwise — shard output (head dim)
            f"{p}.self_attn.q_proj": ColwiseParallel(),
            f"{p}.self_attn.k_proj": ColwiseParallel(),
            f"{p}.self_attn.v_proj": ColwiseParallel(),
            # O: Rowwise — all-reduce output
            f"{p}.self_attn.o_proj": RowwiseParallel(),
            # MLP projections
            f"{p}.mlp.gate_proj": ColwiseParallel(),
            f"{p}.mlp.up_proj": ColwiseParallel(),
            f"{p}.mlp.down_proj": RowwiseParallel(),
        })
    return plan
