"""Tests for H4: GQA repeat_interleave (benchmarked faster than expand+reshape)."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestGQARepeatInterleave:
    """Verify repeat_interleave GQA works correctly in all models."""

    def test_repeat_interleave_correctness(self, device):
        B, num_kv_heads, S, D = 2, 8, 64, 128
        num_groups = 2
        k = torch.randn(B, num_kv_heads, S, D, device=device, dtype=torch.bfloat16)

        k_repeat = k.repeat_interleave(num_groups, dim=1)

        assert k_repeat.shape == (B, num_kv_heads * num_groups, S, D)
        # Each KV head should be repeated num_groups times
        for h in range(num_kv_heads):
            for g in range(num_groups):
                assert torch.equal(k_repeat[:, h * num_groups + g], k[:, h])

    def test_sdpa_handles_repeated_kv(self, device):
        """SDPA works correctly with repeat_interleave KV."""
        B, S, D = 2, 32, 64
        num_q_heads, num_kv_heads = 8, 4
        num_groups = num_q_heads // num_kv_heads

        q = torch.randn(B, num_q_heads, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, num_kv_heads, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, num_kv_heads, S, D, device=device, dtype=torch.bfloat16)

        k_rep = k.repeat_interleave(num_groups, dim=1)
        v_rep = v.repeat_interleave(num_groups, dim=1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        assert out.shape == (B, num_q_heads, S, D)
        assert not out.isnan().any()

    def test_qwen3_model_with_gqa(self, device):
        """Qwen3 model forward works with GQA repeat_interleave."""
        from lset.models.decoder.qwen3.config import Qwen3Config
        from lset.models.decoder.qwen3.model import Qwen3Decoder

        config = Qwen3Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=100,
            max_position_embeddings=64,
        )
        model = Qwen3Decoder(config).to(device=device, dtype=torch.bfloat16)
        ids = torch.randint(0, 100, (2, 16), device=device)
        mask = torch.ones(2, 16, dtype=torch.long, device=device)

        out = model(ids, mask)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert not out["hidden_states"].isnan().any()
