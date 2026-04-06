"""Tests for QLoRA (NF4 quantization + LoRA)."""

import pytest
import torch
import torch.nn as nn

from lset.train.lora import LoRALinear
from lset.train.lora import apply_qlora
from lset.train.lora import get_lora_params
from lset.train.quantization.nf4 import _compute_scaler_block_size


@pytest.fixture
def simple_model():
    """Model with Qwen3-like linear structure."""
    torch.manual_seed(42)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    layer = model.layers[0]
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
    layer.self_attn.k_proj = nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
    layer.self_attn.v_proj = nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
    layer.self_attn.o_proj = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)
    layer.mlp.up_proj = nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)
    layer.mlp.down_proj = nn.Linear(512, 256, bias=False, dtype=torch.bfloat16)
    model.embed_tokens = nn.Embedding(100, 256, dtype=torch.bfloat16)
    model.norm = nn.LayerNorm(256, dtype=torch.bfloat16)
    return model


class TestScalerBlockSize:
    def test_default_works(self):
        # 1024*512=524288, /64=8192, /256=32 -> OK
        assert _compute_scaler_block_size(524288, 64, 256) == 256

    def test_fallback_for_small(self):
        # 256*64=16384, /64=256, /256=1 -> OK
        assert _compute_scaler_block_size(16384, 64, 256) == 256

    def test_fallback_for_odd(self):
        # 128*64=8192, /64=128, 128%256!=0 -> fallback to 128
        sbs = _compute_scaler_block_size(8192, 64, 256)
        assert 8192 // 64 % sbs == 0
        assert sbs > 0


class TestQLoRA:
    def test_apply_qlora_creates_lora_with_nf4(self, simple_model):
        from torchao.dtypes.nf4tensor import NF4Tensor

        apply_qlora(simple_model, r=4, alpha=8.0)
        layer = simple_model.layers[0]
        # Should be LoRALinear wrapping NF4-quantized base
        q_proj = layer.self_attn.q_proj
        assert isinstance(q_proj, LoRALinear)
        assert isinstance(q_proj.base_linear.weight.data, NF4Tensor)

    def test_memory_reduction(self, simple_model):
        """NF4 should reduce parameter memory by roughly 4x."""
        # Calculate original param bytes (just target linears)
        orig_bytes = 0
        for name, p in simple_model.named_parameters():
            if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                orig_bytes += p.numel() * p.element_size()

        apply_qlora(simple_model, r=4, alpha=8.0)

        # Calculate NF4 storage (quantized_data is the main storage)
        from torchao.dtypes.nf4tensor import NF4Tensor

        nf4_bytes = 0
        for name, module in simple_model.named_modules():
            if isinstance(module, LoRALinear):
                w = module.base_linear.weight.data
                if isinstance(w, NF4Tensor):
                    nf4_bytes += w.quantized_data.nbytes
                    nf4_bytes += w.quantized_scalers.nbytes
                    nf4_bytes += w.quantization_factor.nbytes

        ratio = orig_bytes / nf4_bytes
        assert ratio > 3.0, f"Expected >3x compression, got {ratio:.1f}x"

    def test_qlora_forward_not_nan(self, simple_model):
        apply_qlora(simple_model, r=4, alpha=8.0)
        layer = simple_model.layers[0]
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        out = layer.self_attn.q_proj(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_qlora_backward(self, simple_model):
        apply_qlora(simple_model, r=4, alpha=8.0)
        layer = simple_model.layers[0]
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        out = layer.self_attn.q_proj(x)
        out.sum().backward()
        q_proj = layer.self_attn.q_proj
        assert q_proj.lora_A.weight.grad is not None
        assert q_proj.lora_B.weight.grad is not None

    def test_only_lora_trainable(self, simple_model):
        apply_qlora(simple_model, r=4, alpha=8.0)
        lora_params = get_lora_params(simple_model)
        total_params = list(simple_model.parameters())
        assert len(lora_params) > 0
        assert len(lora_params) < len(total_params)
        for p in lora_params:
            assert p.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_qlora_training_steps(self):
        """10 training steps with QLoRA on GPU."""
        torch.manual_seed(42)
        model = nn.Module()
        model.linear = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16, device="cuda")
        apply_qlora(model, r=8, alpha=16.0, target_modules=["linear"])
        model = model.cuda()

        optimizer = torch.optim.AdamW(get_lora_params(model), lr=1e-3)
        losses = []

        for step in range(10):
            x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
            target = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
            out = model.linear(x)
            loss = nn.functional.mse_loss(out.float(), target.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should generally decrease (not strictly, but trend)
        assert losses[-1] < losses[0] * 1.5, f"Loss didn't converge: {losses}"
