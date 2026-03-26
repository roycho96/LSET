"""Tensor Parallel plans for Qwen3 model.

Two plans available:
1. With SequenceParallel (padded mode) — following torchtitan Qwen3 pattern:
   - SP on norms: reduces activation memory by tp_size along sequence dim
   - PrepareModuleInput: Shard(1) → Replicate before attention/MLP
   - RowwiseParallel output_layouts=Shard(1): distribute output back for SP
   - Embedding outputs Shard(1), final norm uses SP

2. Without SequenceParallel (packed mode) — simple plan:
   - ColwiseParallel/RowwiseParallel only
   - Packed tensors are (T, H) with variable T, incompatible with Shard(1) on seq dim
"""

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed._tensor import Replicate, Shard


def get_tp_plan(config, use_sequence_parallel: bool = False) -> dict:
    """Get TP parallelization plan for Qwen3Decoder.

    Args:
        config: Qwen3Config with num_hidden_layers.
        use_sequence_parallel: Enable SequenceParallel on norms. Only for padded mode.

    Returns:
        Dict mapping FQN patterns to ParallelStyle instances.
    """
    if use_sequence_parallel:
        return _get_sp_plan(config)
    return _get_basic_plan(config)


def _get_basic_plan(config) -> dict:
    """Basic TP plan without SequenceParallel — works with both padded and packed."""
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


def _get_sp_plan(config) -> dict:
    """TP plan with SequenceParallel — padded mode only.

    Data flow per block:
      input_layernorm (SP): Shard(1) → Shard(1)
      PrepareModuleInput: Shard(1) → Replicate (all-gather)
      q/k/v_proj (Colwise): Replicate → local shard (heads)
      attention computation: on local tensors
      o_proj (Rowwise, output=Shard(1)): local → Shard(1) (reduce-scatter)
      residual add: Shard(1) + Shard(1) = Shard(1)
      post_attention_layernorm (SP): Shard(1) → Shard(1)
      ... same for MLP ...

    Following torchtitan Qwen3 pattern (torchtitan/models/qwen3/parallelize.py).
    """
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
        plan.update({
            # Block norms: SequenceParallel
            f"{p}.input_layernorm": SequenceParallel(),
            f"{p}.post_attention_layernorm": SequenceParallel(),

            # Attention: convert Shard(1) → Replicate before entering
            # forward(hidden_states, cos, sin, attention_mask) — 4 positional args
            # cos, sin, mask are plain tensors → None (pass through unchanged)
            f"{p}.self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),

            # Q/K/V: Colwise — shard output on head dimension
            f"{p}.self_attn.q_proj": ColwiseParallel(),
            f"{p}.self_attn.k_proj": ColwiseParallel(),
            f"{p}.self_attn.v_proj": ColwiseParallel(),
            # O: Rowwise — reduce-scatter back to Shard(1)
            f"{p}.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),

            # MLP: convert Shard(1) → Replicate
            # forward(x) — 1 positional arg
            f"{p}.mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            f"{p}.mlp.gate_proj": ColwiseParallel(),
            f"{p}.mlp.up_proj": ColwiseParallel(),
            f"{p}.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        })

    return plan
