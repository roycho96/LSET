"""Tests for LogitScale (learnable CLIP-style inverse temperature)."""

import math

import torch

from lset.losses.base import LogitScale


def test_init_scale_matches_exp_log_scale():
    ls = LogitScale(init_scale=20.0, learnable=True)
    assert math.isclose(ls().item(), 20.0, rel_tol=1e-5)


def test_from_temperature_roundtrip():
    ls = LogitScale.from_temperature(0.05, learnable=True)
    assert math.isclose(ls().item(), 20.0, rel_tol=1e-5)


def test_clamp_prevents_overflow():
    ls = LogitScale(init_scale=1.0, max_scale=100.0, learnable=True)
    with torch.no_grad():
        ls.log_scale.copy_(torch.tensor(10.0))  # exp(10) > 100
    assert ls().item() == 100.0


def test_learnable_receives_grad():
    ls = LogitScale(init_scale=10.0, learnable=True)
    s = ls()
    (s * 2.0).backward()
    assert ls.log_scale.grad is not None
    assert ls.log_scale.grad.abs().item() > 0


def test_non_learnable_has_no_grad():
    ls = LogitScale(init_scale=10.0, learnable=False)
    s = ls()
    assert not s.requires_grad
