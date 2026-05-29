"""Tensor Parallel plans for Qwen3 model."""

from torch.distributed._tensor import Replicate
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.parallel import PrepareModuleInput
from torch.distributed.tensor.parallel import RowwiseParallel
from torch.distributed.tensor.parallel import SequenceParallel


def get_tp_plan(
    config, use_sequence_parallel: bool = False, use_lora: bool = False, fused_projections: bool = False
) -> dict:
    """Get TP parallelization plan for Qwen3Decoder."""
    if use_lora:
        return _get_basic_plan(config, use_lora=True)
    if use_sequence_parallel:
        return _get_sp_plan(config, fused_projections=fused_projections)
    return _get_basic_plan(config, fused_projections=fused_projections)


def _get_basic_plan(config, use_lora: bool = False, fused_projections: bool = False) -> dict:
    """Basic TP plan without SequenceParallel — works with both padded and packed."""
    plan = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        if fused_projections and not use_lora:
            colwise = ["self_attn.qkv_proj", "mlp.gate_up_proj"]
        else:
            colwise = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"]
        rowwise = ["self_attn.o_proj", "mlp.down_proj"]

        if use_lora:
            # LoRA wraps nn.Linear → target base_linear + adapters.
            # For Colwise targets: base shards output dim. lora_B shards output to match.
            # For Rowwise targets: base shards input dim. lora_A shards input to match.
            # Non-sharded adapters (lora_A for Colwise, lora_B for Rowwise) work on
            # local tensors since use_local_output=True in the basic plan.
            for name in colwise:
                plan[f"{p}.{name}.base_linear"] = ColwiseParallel()
                plan[f"{p}.{name}.lora_B"] = ColwiseParallel()
            for name in rowwise:
                plan[f"{p}.{name}.base_linear"] = RowwiseParallel()
                plan[f"{p}.{name}.lora_A"] = RowwiseParallel()
        else:
            for name in colwise:
                plan[f"{p}.{name}"] = ColwiseParallel()
            for name in rowwise:
                plan[f"{p}.{name}"] = RowwiseParallel()
    return plan


def _get_sp_plan(config, fused_projections: bool = False) -> dict:
    """TP plan with SequenceParallel — padded mode only."""
    plan = {
        # Model-level: embedding outputs Shard(1) to feed first SP norm
        "embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        # Final norm: SP
        "norm": SequenceParallel(),
    }

    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        plan.update(
            {
                # Block norms: SequenceParallel
                f"{p}.input_layernorm": SequenceParallel(),
                f"{p}.post_attention_layernorm": SequenceParallel(),
                # Attention: convert Shard(1) → Replicate before entering
                f"{p}.self_attn": PrepareModuleInput(
                    input_layouts=(Shard(1), None, None, None),
                    desired_input_layouts=(Replicate(), None, None, None),
                ),
                # MLP: convert Shard(1) → Replicate
                f"{p}.mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
            }
        )

        # Colwise targets
        if fused_projections:
            colwise_names = ["self_attn.qkv_proj", "mlp.gate_up_proj"]
        else:
            colwise_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"]
        # Rowwise targets (with SP output)
        rowwise_sp = {"self_attn.o_proj": Shard(1), "mlp.down_proj": Shard(1)}

        for name in colwise_names:
            plan[f"{p}.{name}"] = ColwiseParallel()
        for name, out_layout in rowwise_sp.items():
            plan[f"{p}.{name}"] = RowwiseParallel(output_layouts=out_layout)

    return plan
