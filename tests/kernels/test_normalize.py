"""Tests for fused L2 normalize kernel."""

import pytest
import torch
import torch.nn.functional as F

from lset.kernels.normalize import _FUSED_NORM_THRESHOLD
from lset.kernels.normalize import fused_l2_normalize
from lset.kernels.normalize import normalize


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestFusedL2Normalize:
    """Tests for the Triton fused L2 normalize kernel."""

    def test_matches_f_normalize_fp32(self, device):
        x = torch.randn(128, 1024, device=device, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = fused_l2_normalize(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_matches_f_normalize_bf16(self, device):
        x = torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
        expected = F.normalize(x, p=2, dim=-1)
        result = fused_l2_normalize(x)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_matches_f_normalize_fp16(self, device):
        x = torch.randn(128, 1024, device=device, dtype=torch.float16)
        expected = F.normalize(x, p=2, dim=-1)
        result = fused_l2_normalize(x)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_output_is_unit_norm(self, device):
        x = torch.randn(256, 768, device=device, dtype=torch.float32)
        result = fused_l2_normalize(x)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_preserves_dtype(self, device):
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            x = torch.randn(64, 512, device=device, dtype=dtype)
            result = fused_l2_normalize(x)
            assert result.dtype == dtype

    def test_small_shape(self, device):
        x = torch.randn(1, 1024, device=device, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = fused_l2_normalize(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_shape(self, device):
        x = torch.randn(8192, 1024, device=device, dtype=torch.bfloat16)
        expected = F.normalize(x, p=2, dim=-1)
        result = fused_l2_normalize(x)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_various_hidden_dims(self, device):
        for D in [768, 1024, 2048, 4096]:
            x = torch.randn(64, D, device=device, dtype=torch.float32)
            expected = F.normalize(x, p=2, dim=-1)
            result = fused_l2_normalize(x)
            assert torch.allclose(result, expected, atol=1e-5), f"Failed for D={D}"

    def test_zero_rows(self, device):
        x = torch.zeros(4, 512, device=device, dtype=torch.float32)
        result = fused_l2_normalize(x)
        # F.normalize returns 0 for zero vectors
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_non_contiguous_input(self, device):
        x = torch.randn(128, 1024, device=device, dtype=torch.float32)
        x_nc = x[:, ::2]  # non-contiguous slice
        expected = F.normalize(x_nc, p=2, dim=-1)
        result = fused_l2_normalize(x_nc)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_gradcheck_fp64(self, device):
        x = torch.randn(4, 32, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda x: fused_l2_normalize(x, eps=1e-12),
            (x,),
            eps=1e-5,
            atol=1e-3,
            fast_mode=True,
        )

    def test_backward_matches_pytorch(self, device):
        x = torch.randn(32, 256, device=device, dtype=torch.float32, requires_grad=True)
        grad = torch.randn(32, 256, device=device, dtype=torch.float32)

        # PyTorch reference
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = F.normalize(x_ref, p=2, dim=-1)
        y_ref.backward(grad)

        # Fused
        y_fused = fused_l2_normalize(x)
        y_fused.backward(grad)

        assert torch.allclose(x.grad, x_ref.grad, atol=1e-5)


class TestNormalizeDispatch:
    """Tests for the threshold-based dispatch function."""

    def test_below_threshold_uses_pytorch(self, device):
        x = torch.randn(64, 1024, device=device, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = normalize(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_above_threshold_uses_fused(self, device):
        x = torch.randn(_FUSED_NORM_THRESHOLD, 1024, device=device, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = normalize(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cpu_always_uses_pytorch(self):
        x = torch.randn(16384, 1024, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = normalize(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_1d_input_uses_pytorch(self, device):
        x = torch.randn(1024, device=device, dtype=torch.float32)
        expected = F.normalize(x, p=2, dim=-1)
        result = normalize(x)
        assert torch.allclose(result, expected, atol=1e-6)
