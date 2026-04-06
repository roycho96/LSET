"""Tests for H3: Attention backend selection."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestAttnBackend:

    def test_backend_setting(self):
        from lset.models.decoder.qwen3.attention import set_attn_backend, get_attn_backend
        set_attn_backend("flash_attn")
        assert get_attn_backend() == "flash_attn"
        set_attn_backend("varlen_attn")
        assert get_attn_backend() == "varlen_attn"
        set_attn_backend("sdpa")
        assert get_attn_backend() == "sdpa"
        set_attn_backend("auto")
        assert get_attn_backend() == "auto"

    def test_invalid_backend_raises(self):
        from lset.models.decoder.qwen3.attention import set_attn_backend
        with pytest.raises(AssertionError):
            set_attn_backend("invalid")

    def test_all_backends_match(self, device):
        """flash_attn, varlen_attn, and SDPA produce similar outputs."""
        from lset.models.decoder.qwen3.attention import (
            _try_flash_attn, _try_varlen_attn, _sdpa_packed_fallback,
        )
        T, H, D = 128, 8, 64
        q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)

        out_fa = _try_flash_attn(q, k, v, cu_seqlens, 64, causal=True)
        out_va = _try_varlen_attn(q, k, v, cu_seqlens, 64, causal=True)
        out_sdpa = _sdpa_packed_fallback(q, k, v, cu_seqlens, 64, H, H, 1, causal=True)

        if out_fa is not None and out_va is not None:
            torch.testing.assert_close(out_fa, out_va, atol=2e-3, rtol=1e-2)
        if out_fa is not None:
            torch.testing.assert_close(out_fa, out_sdpa, atol=2e-3, rtol=1e-2)

    def test_flash_attn_no_graph_break(self, device):
        """flash_attn_varlen_func compiles without graph breaks."""
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            pytest.skip("flash_attn not available")
        from torch._dynamo import explain

        def fn(q, k, v, cu, ms):
            return flash_attn_varlen_func(q, k, v, cu, cu, ms, ms, causal=True)

        T, H, D = 64, 8, 64
        q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)

        explanation = explain(fn)(q, k, v, cu, 32)
        assert explanation.graph_break_count == 0

    def test_varlen_attn_no_graph_break(self, device):
        """varlen_attn compiles without graph breaks."""
        try:
            from torch.nn.attention.varlen import varlen_attn
        except ImportError:
            pytest.skip("varlen_attn not available")
        from torch._dynamo import explain

        def fn(q, k, v, cu, ms):
            return varlen_attn(q, k, v, cu, cu, ms, ms, window_size=(-1, 0))

        T, H, D = 64, 8, 64
        q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)

        explanation = explain(fn)(q, k, v, cu, 32)
        assert explanation.graph_break_count == 0

    def test_bidirectional_varlen(self, device):
        """varlen_attn with causal=False (bidirectional) works."""
        from lset.models.decoder.qwen3.attention import _try_varlen_attn
        T, H, D = 64, 4, 32
        q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)

        out = _try_varlen_attn(q, k, v, cu, 32, causal=False)
        if out is not None:
            assert out.shape == (T, H, D)
            assert not out.isnan().any()

    def test_cpu_fallback_to_sdpa(self):
        """CPU mode falls back to SDPA (no flash_attn/varlen_attn)."""
        from lset.models.decoder.qwen3.attention import _flash_or_sdpa_packed
        T, H, D = 32, 4, 16
        q = torch.randn(T, H, D, dtype=torch.float32)
        k = torch.randn(T, H, D, dtype=torch.float32)
        v = torch.randn(T, H, D, dtype=torch.float32)
        cu = torch.tensor([0, 16, 32], dtype=torch.int32)

        out = _flash_or_sdpa_packed(q, k, v, cu, 16, H, H, 1, causal=True)
        assert out.shape == (T, H, D)
