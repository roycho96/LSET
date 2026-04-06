"""Tests for Tensor Parallel plan and application."""

import torch

from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.parallel_plan import get_tp_plan


def test_tp_basic_plan_keys():
    """Basic TP plan should have 7 entries per layer."""
    config = Qwen3Config(num_hidden_layers=4)
    plan = get_tp_plan(config, use_sequence_parallel=False)
    for i in range(4):
        assert f"layers.{i}.self_attn.q_proj" in plan
        assert f"layers.{i}.self_attn.k_proj" in plan
        assert f"layers.{i}.self_attn.v_proj" in plan
        assert f"layers.{i}.self_attn.o_proj" in plan
        assert f"layers.{i}.mlp.gate_proj" in plan
        assert f"layers.{i}.mlp.up_proj" in plan
        assert f"layers.{i}.mlp.down_proj" in plan
    assert len(plan) == 4 * 7


def test_tp_sp_plan_keys():
    """SP plan should have extra entries for norms, embedding, PrepareModuleInput."""
    from torch.distributed.tensor.parallel import (
        SequenceParallel, PrepareModuleInput, RowwiseParallel,
    )
    config = Qwen3Config(num_hidden_layers=2)
    plan = get_tp_plan(config, use_sequence_parallel=True)

    # Model-level entries
    assert "embed_tokens" in plan
    assert isinstance(plan["embed_tokens"], RowwiseParallel)
    assert "norm" in plan
    assert isinstance(plan["norm"], SequenceParallel)

    # Block-level SP entries
    assert isinstance(plan["layers.0.input_layernorm"], SequenceParallel)
    assert isinstance(plan["layers.0.post_attention_layernorm"], SequenceParallel)
    assert isinstance(plan["layers.0.self_attn"], PrepareModuleInput)
    assert isinstance(plan["layers.0.mlp"], PrepareModuleInput)


def test_tp_plan_correct_parallelism_types():
    """Verify ColwiseParallel for Q/K/V, RowwiseParallel for O/down."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    config = Qwen3Config(num_hidden_layers=2)
    plan = get_tp_plan(config, use_sequence_parallel=False)

    assert isinstance(plan["layers.0.self_attn.q_proj"], ColwiseParallel)
    assert isinstance(plan["layers.0.self_attn.o_proj"], RowwiseParallel)
    assert isinstance(plan["layers.0.mlp.gate_proj"], ColwiseParallel)
    assert isinstance(plan["layers.0.mlp.down_proj"], RowwiseParallel)


def test_attention_local_head_computation():
    """Attention should dynamically compute local heads from output size."""
    from lset.models.decoder.qwen3.attention import Qwen3Attention
    config = Qwen3Config(
        hidden_size=128, num_attention_heads=8,
        num_key_value_heads=4, head_dim=16,
    )
    attn = Qwen3Attention(config)
    x = torch.randn(2, 4, 128)
    cos, sin = torch.ones(1, 1, 4, 16), torch.zeros(1, 1, 4, 16)
    out = attn(x, cos, sin)
    assert out.shape == (2, 4, 128)
