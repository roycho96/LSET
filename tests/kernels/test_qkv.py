"""Tests for fused QKV and GateUp projections."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestFusedQKV:
    """Verify fused QKV output matches separate Q/K/V output with same weights."""

    def test_qwen3_fused_matches_separate(self, device):
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.attention import Qwen3Attention

        config = Qwen3Config(
            hidden_size=64, num_attention_heads=8, num_key_value_heads=4,
            head_dim=16, vocab_size=100, max_position_embeddings=64,
        )
        sep = Qwen3Attention(config, fused_qkv=False).to(device, torch.bfloat16)
        fsd = Qwen3Attention(config, fused_qkv=True).to(device, torch.bfloat16)

        # Copy weights from separate to fused
        with torch.no_grad():
            fsd.qkv_proj.weight.copy_(torch.cat([
                sep.q_proj.weight, sep.k_proj.weight, sep.v_proj.weight
            ], dim=0))
            fsd.o_proj.weight.copy_(sep.o_proj.weight)
            fsd.q_norm.weight.copy_(sep.q_norm.weight)
            fsd.k_norm.weight.copy_(sep.k_norm.weight)

        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        cos = torch.randn(1, 1, 16, 16, device=device, dtype=torch.bfloat16)
        sin = torch.randn(1, 1, 16, 16, device=device, dtype=torch.bfloat16)

        out_sep = sep(x, cos, sin)
        out_fsd = fsd(x, cos, sin)
        torch.testing.assert_close(out_fsd, out_sep, atol=1e-2, rtol=1e-2)

    def test_qwen3_fused_gate_up_matches_separate(self, device):
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.mlp import Qwen3MLP

        config = Qwen3Config(hidden_size=64, intermediate_size=128)
        sep = Qwen3MLP(config, fused_gate_up=False).to(device, torch.bfloat16)
        fsd = Qwen3MLP(config, fused_gate_up=True).to(device, torch.bfloat16)

        with torch.no_grad():
            fsd.gate_up_proj.weight.copy_(torch.cat([
                sep.gate_proj.weight, sep.up_proj.weight
            ], dim=0))
            fsd.down_proj.weight.copy_(sep.down_proj.weight)

        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        out_sep = sep(x)
        out_fsd = fsd(x)
        torch.testing.assert_close(out_fsd, out_sep, atol=1e-2, rtol=1e-2)

    def test_weight_loading_fuses_correctly(self, device):
        """HF separate weights fused into qkv_proj correctly."""
        from lset.models.decoder.qwen3.weights import _fuse_qkv_weights

        # Simulate separate HF weights for 1 layer
        state_dict = {
            "layers.0.self_attn.q_proj.weight": torch.randn(128, 64),
            "layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
            "layers.0.self_attn.v_proj.weight": torch.randn(64, 64),
            "layers.0.self_attn.o_proj.weight": torch.randn(64, 128),
            "layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
            "layers.0.mlp.up_proj.weight": torch.randn(128, 64),
            "layers.0.mlp.down_proj.weight": torch.randn(64, 128),
        }

        fused = _fuse_qkv_weights(state_dict)

        # Check QKV is fused
        assert "layers.0.self_attn.qkv_proj.weight" in fused
        assert "layers.0.self_attn.q_proj.weight" not in fused
        qkv_w = fused["layers.0.self_attn.qkv_proj.weight"]
        assert qkv_w.shape == (256, 64)  # 128 + 64 + 64

        # Check GateUp is fused
        assert "layers.0.mlp.gate_up_proj.weight" in fused
        assert "layers.0.mlp.gate_proj.weight" not in fused
        gu_w = fused["layers.0.mlp.gate_up_proj.weight"]
        assert gu_w.shape == (256, 64)  # 128 + 128

        # Check other weights preserved
        assert "layers.0.self_attn.o_proj.weight" in fused
        assert "layers.0.mlp.down_proj.weight" in fused

    def test_qwen3_model_fused_forward(self, device):
        """Full Qwen3Decoder with fused projections produces valid output."""
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.model import Qwen3Decoder

        config = Qwen3Config(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=8, num_key_value_heads=4, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
        )
        model = Qwen3Decoder(config, fused_projections=True).to(device, torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert not out["hidden_states"].isnan().any()

    def test_qwen3_model_fused_weight_loading(self, device):
        """Load real Qwen3 weights into fused model."""
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.model import Qwen3Decoder
        from lset.models.decoder.qwen3.weights import load_qwen3_weights
        import os

        model_path = os.path.expanduser("~/models/Qwen3-Embedding-0.6B")
        if not os.path.exists(model_path):
            pytest.skip("Qwen3-Embedding-0.6B not available")

        config = Qwen3Config.from_hf_json(f"{model_path}/config.json")
        model = Qwen3Decoder(config, fused_projections=True).to(device, torch.bfloat16)
        state_dict = load_qwen3_weights(model_path, fused_projections=True)
        model.load_state_dict(state_dict, strict=True)

        ids = torch.randint(0, 100, (1, 8), device=device)
        mask = torch.ones(1, 8, dtype=torch.long, device=device)
        out = model(ids, mask)
        assert out["hidden_states"].shape == (1, 8, config.hidden_size)
        assert not out["hidden_states"].isnan().any()

    def test_llama_fused_forward(self, device):
        """LlamaDecoder with fused projections."""
        from lset.models.decoder.llama.config import LlamaConfig
        from lset.models.decoder.llama.model import LlamaDecoder

        config = LlamaConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=8, num_key_value_heads=4, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
        )
        model = LlamaDecoder(config, fused_projections=True).to(device, torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert not out["hidden_states"].isnan().any()

    def test_gemma_fused_forward(self, device):
        """GemmaEmbeddingModel with fused projections."""
        from lset.models.decoder.gemma.config import GemmaConfig
        from lset.models.decoder.gemma.model import GemmaEmbeddingModel

        config = GemmaConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
            query_pre_attn_scalar=16.0,
        )
        model = GemmaEmbeddingModel(config, fused_projections=True).to(device, torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert not out["hidden_states"].isnan().any()

    def test_fused_backward(self, device):
        """Gradients flow through fused projections."""
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.model import Qwen3Decoder

        config = Qwen3Config(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=8, num_key_value_heads=4, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
        )
        model = Qwen3Decoder(config, fused_projections=True).to(device, torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        out["hidden_states"].sum().backward()

        # Check qkv_proj and gate_up_proj have gradients
        for layer in model.layers:
            assert layer.self_attn.qkv_proj.weight.grad is not None
            assert layer.mlp.gate_up_proj.weight.grad is not None
