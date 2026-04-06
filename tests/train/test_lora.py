"""Tests for LoRA module."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from lset.train.lora import (
    LoRALinear,
    apply_lora,
    get_lora_params,
    save_lora_weights,
    load_lora_weights,
)


@pytest.fixture
def base_linear():
    torch.manual_seed(42)
    return nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)


@pytest.fixture
def simple_model():
    """A tiny model mimicking Qwen3 structure for apply_lora testing."""
    torch.manual_seed(42)
    model = nn.Module()
    model.layers = nn.ModuleList([
        nn.Module(),
    ])
    layer = model.layers[0]
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    layer.self_attn.k_proj = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    layer.self_attn.v_proj = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    layer.self_attn.o_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    layer.mlp.up_proj = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    layer.mlp.down_proj = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
    # Non-target modules
    layer.input_layernorm = nn.LayerNorm(64, dtype=torch.bfloat16)
    model.norm = nn.LayerNorm(64, dtype=torch.bfloat16)
    model.embed_tokens = nn.Embedding(100, 64, dtype=torch.bfloat16)
    return model


class TestLoRALinear:
    def test_forward_shape(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_identity_init(self, base_linear):
        """B=zeros means LoRA output starts at zero → same as base."""
        lora = LoRALinear(base_linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        base_out = base_linear(x)
        lora_out = lora(x)
        torch.testing.assert_close(lora_out, base_out, atol=0, rtol=0)

    def test_only_lora_gets_gradients(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        out = lora(x)
        out.sum().backward()
        # Base weight should have no gradient
        assert base_linear.weight.grad is None
        # LoRA weights should have gradients
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_scale_factor(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=16.0)
        assert lora.scale == 4.0

    def test_weight_property(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=8.0)
        assert lora.weight is base_linear.weight

    def test_features_properties(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=8.0)
        assert lora.in_features == 64
        assert lora.out_features == 128

    def test_dropout(self, base_linear):
        lora = LoRALinear(base_linear, r=4, alpha=8.0, dropout=0.1)
        assert isinstance(lora.dropout, nn.Dropout)
        lora_no_drop = LoRALinear(
            nn.Linear(64, 128, bias=False, dtype=torch.bfloat16), r=4, alpha=8.0
        )
        assert isinstance(lora_no_drop.dropout, nn.Identity)


class TestApplyLora:
    def test_replaces_correct_modules(self, simple_model):
        apply_lora(simple_model, r=4, alpha=8.0)
        layer = simple_model.layers[0]
        # All 7 target projections should be LoRALinear
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.k_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        assert isinstance(layer.self_attn.o_proj, LoRALinear)
        assert isinstance(layer.mlp.gate_proj, LoRALinear)
        assert isinstance(layer.mlp.up_proj, LoRALinear)
        assert isinstance(layer.mlp.down_proj, LoRALinear)
        # Non-target modules untouched
        assert isinstance(layer.input_layernorm, nn.LayerNorm)

    def test_freezes_all_base_params(self, simple_model):
        apply_lora(simple_model, r=4, alpha=8.0)
        for name, param in simple_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_custom_targets(self, simple_model):
        apply_lora(simple_model, r=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        layer = simple_model.layers[0]
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        # Not targeted
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
        assert isinstance(layer.self_attn.o_proj, nn.Linear)

    def test_param_count(self, simple_model):
        total_before = sum(p.numel() for p in simple_model.parameters())
        apply_lora(simple_model, r=4, alpha=8.0)
        lora_params = get_lora_params(simple_model)
        trainable = sum(p.numel() for p in lora_params)
        total_after = sum(p.numel() for p in simple_model.parameters())
        # LoRA adds parameters, but trainable is much smaller than total
        assert total_after > total_before
        assert trainable < total_before


class TestGetLoraParams:
    def test_returns_only_trainable(self, simple_model):
        apply_lora(simple_model, r=4, alpha=8.0)
        lora_params = get_lora_params(simple_model)
        assert len(lora_params) > 0
        for p in lora_params:
            assert p.requires_grad


class TestSaveLoadLora:
    def test_roundtrip(self, simple_model):
        apply_lora(simple_model, r=4, alpha=8.0)

        # Modify LoRA weights so they're non-zero
        for name, param in simple_model.named_parameters():
            if "lora_A" in name:
                param.data.fill_(1.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_lora_weights(simple_model, path)

            # Verify saved file exists and has content
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert len(state) > 0
            # All keys should contain lora_A or lora_B
            for key in state:
                assert "lora_A" in key or "lora_B" in key

            # Reset LoRA weights and reload
            for name, param in simple_model.named_parameters():
                if "lora_A" in name:
                    param.data.zero_()

            load_lora_weights(simple_model, path)

            # Verify weights are restored
            for name, param in simple_model.named_parameters():
                if "lora_A" in name:
                    assert (param.data == 1.0).all(), f"{name} not restored"
        finally:
            os.unlink(path)
