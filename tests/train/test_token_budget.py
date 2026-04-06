"""Tests for token-budget chunk planning in GradCache."""

import torch

from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.grad_cache import GradCacheWrapper
from lset.tasks.grad_cache import _plan_chunks_token_budget


class TestPlanChunksTokenBudget:
    def test_uniform_lengths(self):
        lengths = [100] * 10
        chunks = _plan_chunks_token_budget(lengths, budget=300)
        # 3 per chunk at 300 budget, last chunk has 1
        assert all(b < e for b, e in chunks)
        assert chunks[0] == (0, 3)
        assert chunks[-1][1] == 10
        # All sequences covered
        total = sum(e - b for b, e in chunks)
        assert total == 10

    def test_variable_lengths(self):
        lengths = [50, 50, 200, 50, 50]
        chunks = _plan_chunks_token_budget(lengths, budget=150)
        # [50, 50] fits in 150, then [200] alone (exceeds but min 1), then [50, 50]
        total = sum(e - b for b, e in chunks)
        assert total == 5

    def test_single_long_sequence(self):
        lengths = [1000]
        chunks = _plan_chunks_token_budget(lengths, budget=100)
        assert chunks == [(0, 1)]

    def test_all_sequences_included(self):
        lengths = [30, 50, 80, 120, 40, 60, 90, 10]
        chunks = _plan_chunks_token_budget(lengths, budget=200)
        covered = set()
        for b, e in chunks:
            for i in range(b, e):
                covered.add(i)
        assert covered == set(range(len(lengths)))

    def test_budget_roughly_respected(self):
        lengths = [64, 128, 32, 256, 64, 128, 64, 32]
        chunks = _plan_chunks_token_budget(lengths, budget=256)
        for b, e in chunks:
            sum(lengths[b:e])
            # Each chunk should be close to budget (may exceed by one seq)
            # but never by more than the largest sequence in the chunk
            if e - b > 1:
                # Without the last seq, should be under budget
                without_last = sum(lengths[b : e - 1])
                assert without_last <= 256

    def test_empty_input(self):
        chunks = _plan_chunks_token_budget([], budget=100)
        assert chunks == []


class TestGradCacheTokenBudget:
    def _make_packed_data(self, num_seqs=6, min_len=4, max_len=8, H=64):
        lengths = [torch.randint(min_len, max_len + 1, (1,)).item() for _ in range(num_seqs)]
        total = sum(lengths)
        input_ids = torch.randint(0, 100, (total,))
        position_ids = torch.cat([torch.arange(length) for length in lengths])
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0)), dtype=torch.int32)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(lengths),
        }

    def test_token_budget_loss_matches_fixed_chunks(self):
        """Token-budget GradCache produces same loss as fixed-chunk."""
        torch.manual_seed(42)
        config = Qwen3Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=100,
            max_position_embeddings=64,
        )

        task = BiEncoderTask(pooling="last_token", temperature=0.05)

        # Standard forward
        model_std = Qwen3Decoder(config).to(dtype=torch.float32)
        q_batch = self._make_packed_data(num_seqs=4)
        d_batch = self._make_packed_data(num_seqs=4)

        out = task(model_std, q_batch, d_batch)
        std_loss = out["loss"].item()

        # GradCache with token_budget
        model_gc = Qwen3Decoder(config).to(dtype=torch.float32)
        model_gc.load_state_dict(model_std.state_dict())
        gc = GradCacheWrapper(task, chunk_size=16, token_budget=32)
        gc_loss = gc(model_gc, q_batch, d_batch).item()

        assert abs(std_loss - gc_loss) < 1e-4, f"Loss mismatch: {std_loss} vs {gc_loss}"
