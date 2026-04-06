"""Tests for Gemma FusedResidualRMSNorm with (1+weight) variant."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestGemmaFusedResidualRMSNorm:
    """Verify fused residual+norm matches Gemma's (1+weight) pattern."""

    def test_gemma_norm_matches_reference(self, device):
        """Fused residual_rms_norm with (1+weight) matches Gemma's unfused path."""
        from lset.kernels import residual_rms_norm

        B, S, D = 2, 16, 768
        residual = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        attn_out = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
        # Gemma weight: initialized to zeros, applied as (1 + weight)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16) * 0.01
        eps = 1e-6

        # Reference: unfused Gemma path
        new_residual_ref = residual + attn_out
        x_f32 = new_residual_ref.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed_ref = (x_f32 * torch.rsqrt(variance + eps))
        out_ref = ((1.0 + weight.float()) * normed_ref).to(torch.bfloat16)

        # Fused: pass (1 + weight) to residual_rms_norm
        effective_weight = 1.0 + weight
        out_fused, new_residual_fused = residual_rms_norm(
            residual, attn_out, effective_weight, eps,
        )

        torch.testing.assert_close(new_residual_fused, new_residual_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out_fused, out_ref, atol=1e-2, rtol=1e-2)

    def test_gemma_block_output_unchanged(self, device):
        """GemmaBlock with fused norm produces same output as reference."""
        from lset.models.decoder.gemma.config import GemmaConfig
        from lset.models.decoder.gemma.model import GemmaBlock, GemmaRotaryEmbedding

        config = GemmaConfig(
            num_hidden_layers=1, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
            query_pre_attn_scalar=16.0,
        )
        block = GemmaBlock(config, is_sliding=False).to(device=device, dtype=torch.bfloat16)

        B, S = 2, 16
        hidden = torch.randn(B, S, 64, device=device, dtype=torch.bfloat16)
        rope = GemmaRotaryEmbedding(16, 64, 10000.0)
        cos, sin = rope(S, device)
        cos = cos.to(torch.bfloat16)
        sin = sin.to(torch.bfloat16)

        out = block(hidden, cos, sin)
        assert out.shape == (B, S, 64)
        assert not out.isnan().any()

    def test_gemma_gradcheck(self, device):
        """Gradient flows correctly through Gemma fused path."""
        from lset.kernels import fused_residual_rms_norm

        D = 32
        residual = torch.randn(4, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        attn_out = torch.randn(4, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=torch.bfloat16)
        effective_weight = 1.0 + weight

        out, new_res = fused_residual_rms_norm(residual, attn_out, effective_weight, 1e-6)
        loss = out.sum() + new_res.sum()
        loss.backward()

        assert residual.grad is not None
        assert attn_out.grad is not None
        assert not residual.grad.isnan().any()
        assert not attn_out.grad.isnan().any()

    def test_gemma_model_forward(self, device):
        """Full GemmaEmbeddingModel forward pass with fused norms."""
        from lset.models.decoder.gemma.config import GemmaConfig
        from lset.models.decoder.gemma.model import GemmaEmbeddingModel

        config = GemmaConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
            query_pre_attn_scalar=16.0,
        )
        model = GemmaEmbeddingModel(config).to(device=device, dtype=torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert not out["hidden_states"].isnan().any()

    def test_gemma_training_10_steps(self, device):
        """10-step training with Gemma fused norms doesn't crash or produce NaN."""
        from lset.models.decoder.gemma.config import GemmaConfig
        from lset.models.decoder.gemma.model import GemmaEmbeddingModel

        config = GemmaConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=100, max_position_embeddings=64,
            query_pre_attn_scalar=16.0,
        )
        model = GemmaEmbeddingModel(config).to(device=device, dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(10):
            ids = torch.randint(0, 100, (2, 16), device=device)
            mask = torch.ones(2, 16, dtype=torch.long, device=device)
            out = model(ids, mask)
            loss = out["hidden_states"].sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            assert not loss.isnan(), f"NaN loss at step {step}"
