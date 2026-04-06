"""Tests for fused contrastive loss kernel."""

import pytest
import torch
import torch.nn.functional as F

from lset.kernels.loss import fused_dense_loss
from lset.kernels.loss import should_use_fused
from lset.losses.contrastive import contrastive_loss
from lset.losses.fused_contrastive import fused_contrastive_loss


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_embeddings(Q, K, D, device, dtype=torch.float32):
    """Create random normalized embeddings and a label matrix."""
    q = F.normalize(torch.randn(Q, D, device=device, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(K, D, device=device, dtype=dtype), dim=-1)
    return q, k


def _make_labels(Q, K, num_pos_per_query=1, device="cuda"):
    """Create int8 labels with num_pos_per_query positives per query.

    Layout: for query i, docs [i*num_pos_per_query : (i+1)*num_pos_per_query] are positive.
    All other docs are negative (0).
    """
    labels = torch.zeros(Q, K, dtype=torch.int8, device=device)
    for i in range(Q):
        for j in range(num_pos_per_query):
            doc_idx = i * num_pos_per_query + j
            if doc_idx < K:
                labels[i, doc_idx] = 1
    return labels


def _extract_pos_info(labels):
    """Extract pos_qi, pos_di, pos_counts from label matrix."""
    pos_mask = labels > 0
    pos_qi, pos_di = torch.where(pos_mask)
    pos_counts = pos_mask.sum(dim=1).long()
    return pos_qi, pos_di, pos_counts


class TestFusedLossNumericalMatch:
    """Verify fused loss matches reference contrastive_loss."""

    def test_multi_loss_matches_reference(self, device):
        """MP-NCE fused loss matches reference implementation."""
        Q, K, D = 32, 64, 128
        q, k = _make_embeddings(Q, K, D, device)
        labels_int8 = _make_labels(Q, K, num_pos_per_query=1, device=device)
        labels_float = labels_int8.float()
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels_int8)

        # Reference: contrastive_loss
        ref_loss = contrastive_loss(q, k, labels_float, temperature=0.05)

        # Fused: direct kernel call (force fused even though Q*K is small)
        scale = 1.0 / 0.05
        fused_loss = fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )

        assert abs(ref_loss.item() - fused_loss.item()) < 0.05, (
            f"Loss mismatch: ref={ref_loss.item():.4f} fused={fused_loss.item():.4f}"
        )

    def test_multi_loss_multi_positive(self, device):
        """MP-NCE with multiple positives per query."""
        Q, K, D = 16, 64, 128
        q, k = _make_embeddings(Q, K, D, device)
        labels_int8 = _make_labels(Q, K, num_pos_per_query=3, device=device)
        labels_float = labels_int8.float()
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels_int8)

        ref_loss = contrastive_loss(q, k, labels_float, temperature=0.05)
        scale = 1.0 / 0.05
        fused_loss = fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )

        assert abs(ref_loss.item() - fused_loss.item()) < 0.1, (
            f"Loss mismatch: ref={ref_loss.item():.4f} fused={fused_loss.item():.4f}"
        )

    def test_dispatch_below_threshold_uses_reference(self, device):
        """Below threshold, fused_contrastive_loss falls back to reference."""
        Q, K, D = 8, 16, 64
        q, k = _make_embeddings(Q, K, D, device)
        labels = torch.zeros(Q, K, device=device)
        for i in range(Q):
            labels[i, i % K] = 1.0
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels.to(torch.int8))

        # This should use reference path since Q=8 < 1024
        loss = fused_contrastive_loss(
            q, k, labels, temperature=0.05, pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )
        ref = contrastive_loss(q, k, labels, temperature=0.05)
        assert torch.allclose(loss, ref, atol=1e-5)


class TestFusedLossGradient:
    """Gradient correctness tests."""

    def test_backward_produces_gradients(self, device):
        """Fused loss backward produces gradients for q and k."""
        Q, K, D = 32, 64, 128
        q = F.normalize(torch.randn(Q, D, device=device), dim=-1).detach().requires_grad_(True)
        k = F.normalize(torch.randn(K, D, device=device), dim=-1).detach().requires_grad_(True)
        labels_int8 = _make_labels(Q, K, device=device)
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels_int8)

        scale = 1.0 / 0.05
        loss = fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()

    def test_backward_bf16(self, device):
        """Gradient works with bf16 embeddings."""
        Q, K, D = 32, 64, 128
        q = F.normalize(torch.randn(Q, D, device=device, dtype=torch.bfloat16), dim=-1).requires_grad_(True)
        k = F.normalize(torch.randn(K, D, device=device, dtype=torch.bfloat16), dim=-1).requires_grad_(True)
        labels_int8 = _make_labels(Q, K, device=device)
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels_int8)

        scale = 1.0 / 0.05
        loss = fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


class TestFusedLossMemory:
    """Memory usage tests."""

    def test_no_score_matrix_materialized(self, device):
        """Fused path should NOT allocate Q*K*4 bytes for score matrix.

        Run twice: first call triggers Triton JIT/autotune allocation,
        second call measures actual kernel memory usage.
        """
        Q, K, D = 2048, 4096, 1024
        q, k = _make_embeddings(Q, K, D, device, dtype=torch.bfloat16)
        labels_int8 = _make_labels(Q, K, device=device)
        pos_qi, pos_di, pos_counts = _extract_pos_info(labels_int8)
        scale = 1.0 / 0.05

        # Warmup: first call compiles Triton kernels + autotune
        _ = fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )
        torch.cuda.synchronize()

        # Second call: measure actual memory usage
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.max_memory_allocated()

        fused_dense_loss(
            q, k, labels_int8, scale=scale, loss_type="multi", pos_qi=pos_qi, pos_di=pos_di, pos_counts=pos_counts
        )
        torch.cuda.synchronize()

        after = torch.cuda.max_memory_allocated()
        delta_mb = (after - before) / 1024**2

        # Score matrix would be Q*K*4 = 2048*4096*4 = 32MB (fp32)
        score_matrix_mb = Q * K * 4 / 1024**2
        print(f"Memory delta: {delta_mb:.1f}MB, score matrix would be: {score_matrix_mb:.1f}MB")
        assert delta_mb < score_matrix_mb, f"Fused used {delta_mb:.1f}MB, score matrix would be {score_matrix_mb:.1f}MB"


class TestShouldUseFused:
    """Threshold dispatch logic tests."""

    def test_large_q_always_fused(self):
        assert should_use_fused(2048, 100, "multi") is True

    def test_medium_q_needs_large_k(self):
        assert should_use_fused(1024, 4096, "multi") is True
        assert should_use_fused(1024, 1024, "multi") is False

    def test_small_q_never_fused(self):
        assert should_use_fused(512, 8192, "multi") is False

    def test_soft_requires_more(self):
        assert should_use_fused(2048, 2048, "soft") is False
        assert should_use_fused(2048, 4096, "soft") is True

    def test_cross_same_as_soft(self):
        assert should_use_fused(2048, 4096, "cross") is True
        assert should_use_fused(1024, 8192, "cross") is False


class TestFusedLossIntegration:
    """Integration with collator and bi_encoder."""

    def test_collator_emits_pos_info(self):
        """EmbeddingCollator should emit pos_qi, pos_di, pos_counts."""
        from unittest.mock import MagicMock

        from lset.train.data.collator import EmbeddingCollator

        tok = MagicMock()
        tok.encode = lambda text: MagicMock(ids=list(range(len(text))))

        collator = EmbeddingCollator(tok, max_length=512)
        batch = [
            {"query": "hello", "positives": ["pos1", "pos2"], "negatives": ["neg1"]},
            {"query": "world", "positives": ["pos3"], "negatives": []},
        ]
        result = collator(batch)

        assert "pos_qi" in result
        assert "pos_di" in result
        assert "pos_counts" in result
        assert result["pos_qi"].shape[0] == 3  # 2 + 1 positives
        assert result["pos_di"].shape[0] == 3
        assert result["pos_counts"].tolist() == [2, 1]

    def test_pos_info_matches_labels(self):
        """pos_qi/pos_di should match torch.where(labels > 0)."""
        from unittest.mock import MagicMock

        from lset.train.data.collator import EmbeddingCollator

        tok = MagicMock()
        tok.encode = lambda text: MagicMock(ids=list(range(len(text))))

        collator = EmbeddingCollator(tok, max_length=512)
        batch = [
            {"query": "q1", "positives": ["p1"], "negatives": ["n1"]},
            {"query": "q2", "positives": ["p2", "p3"], "negatives": []},
        ]
        result = collator(batch)

        labels = result["labels"]
        expected_qi, expected_di = torch.where(labels > 0)

        assert torch.equal(result["pos_qi"], expected_qi)
        assert torch.equal(result["pos_di"], expected_di)
