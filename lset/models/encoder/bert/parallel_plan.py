"""Tensor Parallel plan for BERT/XLM-RoBERTa encoder."""

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def get_tp_plan(config, **kwargs) -> dict:
    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update({
            f"{p}.attention.query": ColwiseParallel(),
            f"{p}.attention.key": ColwiseParallel(),
            f"{p}.attention.value": ColwiseParallel(),
            f"{p}.attention.dense": RowwiseParallel(),
            f"{p}.mlp.dense_in": ColwiseParallel(),
            f"{p}.mlp.dense_out": RowwiseParallel(),
        })
    return plan
