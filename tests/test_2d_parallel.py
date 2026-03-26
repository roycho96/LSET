"""Tests for 2D parallelism (TP + FSDP2).

These tests require 2 GPUs and are run with torchrun.
They test TP-only (tp=2, dp=1) on 2 GPUs.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_2d_parallel.py -v
"""

import os
import pytest
import torch
import torch.distributed as dist

from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.model import Qwen3Decoder
from lset.models import get_model_spec
from lset.distributed.parallel import build_parallel_model, ParallelConfig
from lset.distributed.mesh import build_mesh


def _is_distributed():
    return os.environ.get("RANK") is not None


requires_distributed = pytest.mark.skipif(
    not _is_distributed(),
    reason="Requires torchrun with 2+ GPUs"
)


@requires_distributed
def test_tp_forward_and_training():
    """TP forward pass and training loop should run without error."""
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

    pconfig = ParallelConfig(dp_size=1, tp_size=2)
    model, mesh = build_parallel_model(model, config, pconfig)

    # Forward pass test
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[1, 2, 3, 4]], device="cuda")
        mask = torch.ones(1, 4, device="cuda", dtype=torch.long)
        out = model(x, mask)
        assert "hidden_states" in out
        assert out["hidden_states"].shape[0] == 1

    # Training test
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    for i in range(10):
        x = torch.randint(0, 1000, (2, 16), device="cuda")
        mask = torch.ones(2, 16, device="cuda", dtype=torch.long)
        out = model(x, mask)
        loss = out["hidden_states"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Verify loss is not NaN
    assert not torch.isnan(loss), "Loss should not be NaN"

    dist.destroy_process_group()
