"""Tests for plan_minibatches + MinibatchBackward dispatch."""

import torch

from lset.train.grad_cache import BasicMinibatchBackward
from lset.train.grad_cache import DDPMinibatchBackward
from lset.train.grad_cache import DeepSpeedMinibatchBackward
from lset.train.grad_cache import FSDP2MinibatchBackward
from lset.train.grad_cache import MinibatchBackward
from lset.train.grad_cache import plan_minibatches


def test_plan_padded_fixed_size():
    feat = {"input_ids": torch.zeros(10, 16, dtype=torch.long), "attention_mask": torch.ones(10, 16)}
    assert plan_minibatches(feat, mini_batch_size=4) == [(0, 4), (4, 8), (8, 10)]


def test_plan_packed_fixed_size():
    cu = torch.tensor([0, 8, 16, 24, 32, 40])  # 5 sequences
    feat = {"input_ids": torch.zeros(40, dtype=torch.long), "cu_seqlens": cu}
    assert plan_minibatches(feat, mini_batch_size=2) == [(0, 2), (2, 4), (4, 5)]


def test_plan_packed_token_budget_long_seqs():
    cu = torch.tensor([0, 10, 20, 30, 40])  # 4 seqs of 10 tokens each
    feat = {"input_ids": torch.zeros(40, dtype=torch.long), "cu_seqlens": cu}
    plan = plan_minibatches(feat, mini_batch_size=100, token_budget=15)
    # 10+10 > 15 so each bin has 1 seq: every seq is its own minibatch
    assert plan == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_plan_packed_token_budget_packs_small_seqs():
    cu = torch.tensor([0, 3, 6, 9, 12])
    feat = {"input_ids": torch.zeros(12, dtype=torch.long), "cu_seqlens": cu}
    plan = plan_minibatches(feat, mini_batch_size=100, token_budget=7)
    # 3+3=6 fits, +3=9 doesn't → split. Expect [(0,2),(2,4)]
    assert plan == [(0, 2), (2, 4)]


def test_for_model_defaults_to_basic():
    rt = MinibatchBackward.for_model(torch.nn.Linear(4, 4))
    assert isinstance(rt, BasicMinibatchBackward)


def test_align_minibatches_noop_single_rank():
    rt = BasicMinibatchBackward()
    plans = [[(0, 8), (8, 12)], [(0, 10)]]
    aligned, num_valid = rt.align_minibatches(plans)
    assert aligned == plans
    assert num_valid == [2, 1]


def test_basic_backward_triggers_autograd():
    rt = BasicMinibatchBackward()
    x = torch.randn(4, requires_grad=True)
    surrogate = (x * 2.0).sum()
    with rt.context():
        rt.backward(surrogate, is_last=True)
    rt.finalize([x])
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))


def test_deepspeed_runtime_duck_typed():
    """Fake DS engine confirms our duck-typed dispatcher routes correctly."""

    class FakeEngine:
        def __init__(self):
            self.module = torch.nn.Linear(4, 4)
            self.backward_calls = []
            self.boundary_calls = []

        def set_gradient_accumulation_boundary(self, flag):
            self.boundary_calls.append(flag)

        def backward(self, loss):
            self.backward_calls.append(float(loss.item()))

    engine = FakeEngine()
    rt = MinibatchBackward.for_model(engine, is_ga_boundary=True)
    assert isinstance(rt, DeepSpeedMinibatchBackward)

    x = torch.ones(())
    rt.backward(x * 1.0, is_last=False)
    rt.backward(x * 2.0, is_last=True)

    assert engine.boundary_calls == [False, True]
    assert engine.backward_calls == [1.0, 2.0]


def test_fsdp2_runtime_detected():
    class FakeFSDP:
        def __init__(self):
            self.sync_history = []

        def set_requires_gradient_sync(self, flag):
            self.sync_history.append(flag)

        def set_reshard_after_backward(self, flag):
            pass

    mod = FakeFSDP()
    rt = MinibatchBackward.for_model(mod, is_ga_boundary=True)
    assert isinstance(rt, FSDP2MinibatchBackward)

    with rt.context():
        # sync disabled on enter
        assert mod.sync_history[-1] is False
    # sync re-enabled on exit
    assert mod.sync_history[-1] is True
