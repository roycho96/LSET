"""Tests for Triton packed segment mean pooling kernel."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _reference_segment_mean(hidden, cu_seqlens, normalize=False):
    """Reference implementation using scatter_add_."""
    M = cu_seqlens.shape[0] - 1
    H = hidden.shape[-1]
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    seq_ids = torch.repeat_interleave(
        torch.arange(M, device=hidden.device), lengths,
    )
    emb = torch.zeros(M, H, dtype=hidden.dtype, device=hidden.device)
    emb.scatter_add_(0, seq_ids.unsqueeze(-1).expand_as(hidden), hidden)
    emb = emb / lengths.unsqueeze(-1).to(emb.dtype).clamp(min=1e-9)
    if normalize:
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


class TestTritonSegmentMean:
    """Numerical correctness vs scatter_add_ reference."""

    def test_mean_matches_reference(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        T, H = 256, 1024
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 50, 120, 180, 256], dtype=torch.int32, device=device)

        ref = _reference_segment_mean(hidden, cu_seqlens, normalize=False)
        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=False)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_mean_normalize_matches_reference(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        T, H = 256, 1024
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 64, 128, 256], dtype=torch.int32, device=device)

        ref = _reference_segment_mean(hidden, cu_seqlens, normalize=True)
        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=True)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_single_sequence(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        T, H = 32, 128
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=False)
        ref = hidden.float().mean(dim=0, keepdim=True).to(torch.bfloat16)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_length_one_sequences(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        H = 64
        hidden = torch.randn(5, H, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)

        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=False)
        # Each sequence has 1 token, so mean = the token itself
        torch.testing.assert_close(out, hidden, atol=1e-3, rtol=1e-3)

    def test_gradient_flows(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        T, H = 64, 128
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
        cu_seqlens = torch.tensor([0, 20, 40, 64], dtype=torch.int32, device=device)

        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=False)
        out.sum().backward()

        assert hidden.grad is not None
        assert not hidden.grad.isnan().any()
        # Each token's grad should be 1/length for its segment
        for i in range(3):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            length = e - s
            expected_grad = 1.0 / length
            actual = hidden.grad[s:e].mean().item()
            assert abs(actual - expected_grad) < 0.01, \
                f"Segment {i}: expected grad ~{expected_grad:.4f}, got {actual:.4f}"

    def test_gradient_normalize(self, device):
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        T, H = 64, 128
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
        cu_seqlens = torch.tensor([0, 30, 64], dtype=torch.int32, device=device)

        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=True)
        out.sum().backward()

        assert hidden.grad is not None
        assert not hidden.grad.isnan().any()

    def test_many_sequences(self, device):
        """500 sequences with variable lengths."""
        from lset.kernels.triton_segment_pool import triton_segment_mean_pool

        H = 1024
        torch.manual_seed(42)
        lengths = torch.randint(10, 100, (500,))
        T = int(lengths.sum())
        cu = torch.zeros(501, dtype=torch.int32)
        cu[1:] = lengths.cumsum(0).int()
        cu_seqlens = cu.to(device)

        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        ref = _reference_segment_mean(hidden, cu_seqlens, normalize=True)
        out = triton_segment_mean_pool(hidden, cu_seqlens, normalize=True)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_packed_pool_dispatcher_uses_triton(self, device):
        """packed_pool routes mean pooling to Triton kernel on CUDA."""
        from lset.tasks.packed_pooling import packed_pool

        T, H = 128, 256
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 40, 80, 128], dtype=torch.int32, device=device)

        # Should use Triton path (on CUDA, mean strategy)
        out = packed_pool(hidden, cu_seqlens, strategy="mean", normalize=True)
        assert out.shape == (3, H)
        assert not out.isnan().any()

        # Verify it's normalized
        norms = out.float().norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-2, rtol=1e-2)
