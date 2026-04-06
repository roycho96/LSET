"""Tests for 2D parallelism (TP + FSDP2).

These tests require 2 GPUs and are run with torchrun.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_2d_parallel.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from lset.distributed.parallel import ParallelConfig
from lset.distributed.parallel import build_parallel_model
from lset.models import get_model_spec


def _is_distributed():
    return os.environ.get("RANK") is not None


requires_distributed = pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 2+ GPUs")


@requires_distributed
def test_tp_with_sequence_parallel():
    """TP + SequenceParallel forward + training on 2 GPUs."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    model_path = os.path.expanduser("~/models/Qwen3-Embedding-0.6B")
    spec = get_model_spec("qwen3")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)

    # Test with SP enabled (padded mode)
    pconfig = ParallelConfig(dp_size=1, tp_size=2, use_sequence_parallel=True)
    model, mesh = build_parallel_model(model, config, pconfig)

    # Forward pass — verify output is a regular tensor (not DTensor)
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[1, 2, 3, 4]], device="cuda")
        mask = torch.ones(1, 4, device="cuda", dtype=torch.long)
        out = model(x, mask)
        hs = out["hidden_states"]
        assert hs.shape == (1, 4, config.hidden_size)
        # Verify it's a plain tensor, not DTensor (full_tensor was called)
        assert type(hs) is torch.Tensor, f"Expected plain Tensor, got {type(hs)}"

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    losses = []
    for i in range(10):
        x = torch.randint(0, 1000, (2, 16), device="cuda")
        mask = torch.ones(2, 16, device="cuda", dtype=torch.long)
        out = model(x, mask)
        loss = out["hidden_states"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert all(not torch.isnan(torch.tensor(v)) for v in losses), "NaN detected"

    # Measure memory savings from SP
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 1000, (4, 128), device="cuda")
    mask = torch.ones(4, 128, device="cuda", dtype=torch.long)
    out = model(x, mask)
    out["hidden_states"].sum().backward()
    peak_sp = torch.cuda.max_memory_allocated() / 1024 / 1024
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"SP peak memory (4x128 forward+backward): {peak_sp:.1f} MB")

    dist.destroy_process_group()
