"""Tensor Parallel plan for Llama — same structure as Qwen3 basic plan."""

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def get_tp_plan(config, **kwargs) -> dict:
    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update({
            f"{p}.self_attn.q_proj": ColwiseParallel(),
            f"{p}.self_attn.k_proj": ColwiseParallel(),
            f"{p}.self_attn.v_proj": ColwiseParallel(),
            f"{p}.self_attn.o_proj": RowwiseParallel(),
            f"{p}.mlp.gate_proj": ColwiseParallel(),
            f"{p}.mlp.up_proj": ColwiseParallel(),
            f"{p}.mlp.down_proj": RowwiseParallel(),
        })
    return plan
