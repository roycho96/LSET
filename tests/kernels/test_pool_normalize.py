"""Tests for H2: FusedPoolNormScore (fused pooling + L2 normalize)."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_packed_data(num_seqs=4, min_len=8, max_len=32, D=256, device="cuda", dtype=torch.bfloat16):
    """Create packed hidden states with random sequence lengths."""
    lengths = torch.randint(min_len, max_len + 1, (num_seqs,))
    T = lengths.sum().item()
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = lengths.cumsum(0)
    hidden = torch.randn(T, D, device=device, dtype=dtype)
    return hidden, cu_seqlens, lengths


def _reference_pool_normalize(hidden, cu_seqlens, strategy, eps=1e-12):
    """Reference: separate pool then normalize."""
    num_seqs = cu_seqlens.shape[0] - 1
    D = hidden.shape[-1]

    if strategy == "last_token":
        indices = (cu_seqlens[1:] - 1).long()
        emb = hidden[indices]
    elif strategy == "cls":
        indices = cu_seqlens[:-1].long()
        emb = hidden[indices]
    elif strategy == "mean":
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden.device),
            lengths,
        )
        emb = torch.zeros(num_seqs, D, dtype=hidden.dtype, device=hidden.device)
        emb.scatter_add_(0, seq_ids.unsqueeze(-1).expand_as(hidden), hidden)
        emb = emb / lengths.unsqueeze(-1).to(emb.dtype).clamp(min=1e-9)
    return F.normalize(emb.float(), p=2, dim=-1).to(hidden.dtype)


class TestFusedPoolNormalize:
    @pytest.mark.parametrize("strategy", ["last_token", "cls", "mean"])
    def test_numerical_match_bf16(self, device, strategy):
        from lset.kernels.pool_normalize import fused_pool_normalize

        hidden, cu_seqlens, _ = _make_packed_data(device=device)

        out_fused = fused_pool_normalize(hidden, cu_seqlens, strategy)
        out_ref = _reference_pool_normalize(hidden, cu_seqlens, strategy)

        torch.testing.assert_close(out_fused, out_ref, atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("strategy", ["last_token", "cls", "mean"])
    def test_output_is_unit_norm(self, device, strategy):
        from lset.kernels.pool_normalize import fused_pool_normalize

        hidden, cu_seqlens, _ = _make_packed_data(device=device, dtype=torch.float32)

        out = fused_pool_normalize(hidden, cu_seqlens, strategy)
        norms = out.float().norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("strategy", ["last_token", "cls"])
    def test_backward_gather_normalize(self, device, strategy):
        from lset.kernels.pool_normalize import fused_pool_normalize

        hidden, cu_seqlens, _ = _make_packed_data(device=device, dtype=torch.float32)
        hidden.requires_grad_(True)

        out = fused_pool_normalize(hidden, cu_seqlens, strategy)
        out.sum().backward()
        assert hidden.grad is not None
        assert not hidden.grad.isnan().any()

    def test_backward_mean(self, device):
        from lset.kernels.pool_normalize import fused_pool_normalize

        hidden, cu_seqlens, _ = _make_packed_data(device=device, dtype=torch.float32)
        hidden.requires_grad_(True)

        out = fused_pool_normalize(hidden, cu_seqlens, "mean")
        out.sum().backward()
        assert hidden.grad is not None

    def test_packed_pool_integration(self, device):
        """packed_pool uses fused path on CUDA."""
        from lset.tasks.packed_pooling import packed_pool

        hidden, cu_seqlens, _ = _make_packed_data(device=device)

        emb = packed_pool(hidden, cu_seqlens, "last_token", normalize=True)
        assert emb.shape == (cu_seqlens.shape[0] - 1, hidden.shape[-1])
        norms = emb.float().norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-2, rtol=1e-2)

    def test_single_sequence(self, device):
        from lset.kernels.pool_normalize import fused_pool_normalize

        T, D = 16, 128
        hidden = torch.randn(T, D, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        out = fused_pool_normalize(hidden, cu_seqlens, "last_token")
        assert out.shape == (1, D)
