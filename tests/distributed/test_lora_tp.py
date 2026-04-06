"""Tests for LoRA + Tensor Parallelism.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_lora_tp.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from lset.distributed.parallel import ParallelConfig
from lset.distributed.parallel import build_parallel_model
from lset.models import get_model_spec
from lset.train.lora import apply_lora
from lset.train.lora import get_lora_params


def _is_distributed():
    return os.environ.get("RANK") is not None


requires_distributed = pytest.mark.skipif(
    not _is_distributed(),
    reason="Requires torchrun with 2+ GPUs",
)


def _build_model():
    model_path = os.path.expanduser("~/models/Qwen3-Embedding-0.6B")
    spec = get_model_spec("qwen3")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    return model, config


def _train_steps(model, n_steps=10, lr=1e-4):
    lora_params = get_lora_params(model)
    assert len(lora_params) > 0, "No LoRA params found"
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    model.train()
    losses = []
    for step in range(n_steps):
        x = torch.randint(0, 1000, (2, 16), device="cuda")
        mask = torch.ones(2, 16, device="cuda", dtype=torch.long)
        out = model(x, mask)
        loss = out["hidden_states"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


@requires_distributed
def test_lora_tp_training():
    """LoRA + TP training 10 steps on 2 GPUs."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    model, config = _build_model()
    apply_lora(model, r=8, alpha=16.0)

    pconfig = ParallelConfig(dp_size=1, tp_size=2, use_lora=True)
    model, mesh = build_parallel_model(model, config, pconfig)

    losses = _train_steps(model)
    assert all(not torch.isnan(torch.tensor(v)) for v in losses), "NaN in losses"
    if rank == 0:
        print(f"LoRA+TP losses: {losses[0]:.2f} → {losses[-1]:.2f}")

    dist.destroy_process_group()
