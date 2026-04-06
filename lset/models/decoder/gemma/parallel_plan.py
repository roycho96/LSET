"""Tensor Parallel plan for EmbeddingGemma."""

from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.parallel import RowwiseParallel


def get_tp_plan(config, **kwargs) -> dict:
    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update(
            {
                f"{p}.self_attn.q_proj": ColwiseParallel(),
                f"{p}.self_attn.k_proj": ColwiseParallel(),
                f"{p}.self_attn.v_proj": ColwiseParallel(),
                f"{p}.self_attn.o_proj": RowwiseParallel(),
                f"{p}.mlp.gate_proj": ColwiseParallel(),
                f"{p}.mlp.up_proj": ColwiseParallel(),
                f"{p}.mlp.down_proj": RowwiseParallel(),
            }
        )
    return plan
