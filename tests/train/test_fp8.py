"""Tests for Float8 (FP8) training.

Run single-GPU tests:
    python -m pytest tests/test_fp8_training.py -v -k "not distributed"

Run 2-GPU tests:
    torchrun --nproc_per_node=2 -m pytest tests/test_fp8_training.py -v -k "distributed"
"""

import os

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def fp8_model():
    """Create a simple model suitable for FP8 (dims divisible by 16)."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda"),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False, dtype=torch.bfloat16, device="cuda"),
    )
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFP8Training:
    def test_apply_fp8_training(self, fp8_model):
        from lset.distributed.fp8 import apply_fp8_training
        apply_fp8_training(fp8_model, recipe="rowwise")
        # Check Float8Linear was applied
        from torchao.float8.float8_linear import Float8Linear
        assert isinstance(fp8_model[0], Float8Linear)
        assert isinstance(fp8_model[2], Float8Linear)

    def test_fp8_forward_valid(self, fp8_model):
        from lset.distributed.fp8 import apply_fp8_training
        apply_fp8_training(fp8_model, recipe="rowwise")
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        out = fp8_model(x)
        assert out.shape == (16, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_fp8_backward(self, fp8_model):
        from lset.distributed.fp8 import apply_fp8_training
        apply_fp8_training(fp8_model, recipe="rowwise")
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        out = fp8_model(x)
        out.sum().backward()
        assert fp8_model[0].weight.grad is not None

    def test_fp8_training_steps_with_compile(self, fp8_model):
        """10 training steps with FP8 + torch.compile."""
        from lset.distributed.fp8 import apply_fp8_training
        apply_fp8_training(fp8_model, recipe="rowwise")
        model_c = torch.compile(fp8_model)
        optimizer = torch.optim.AdamW(model_c.parameters(), lr=1e-3)

        losses = []
        for step in range(10):
            x = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
            target = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
            out = model_c(x)
            loss = nn.functional.mse_loss(out.float(), target.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert all(not torch.isnan(torch.tensor(l)) for l in losses), f"NaN: {losses}"

    def test_fp8_invalid_recipe(self, fp8_model):
        from lset.distributed.fp8 import apply_fp8_training
        with pytest.raises(ValueError, match="Invalid FP8 recipe"):
            apply_fp8_training(fp8_model, recipe="invalid")

    def test_fp8_skips_small_dims(self):
        """Linear with dims not divisible by 16 should be skipped."""
        from lset.distributed.fp8 import apply_fp8_training
        from torchao.float8.float8_linear import Float8Linear
        model = nn.Sequential(
            nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda"),
            nn.Linear(256, 13, bias=False, dtype=torch.bfloat16, device="cuda"),  # 13 not div by 16
        )
        apply_fp8_training(model, recipe="rowwise")
        assert isinstance(model[0], Float8Linear)
        assert isinstance(model[1], nn.Linear)  # Skipped

    def test_fp8_lora_error(self):
        """FP8 + LoRA should error in engine."""
        from lset.train.engine import TrainingEngine
        with pytest.raises(ValueError, match="FP8 training \\+ LoRA is not supported"):
            TrainingEngine(
                model_name="qwen3",
                model_path="dummy",
                dataset=None,
                fp8=True,
                lora=True,
            )

    def test_fp8_memory_vs_bf16(self):
        """FP8 should not use significantly more memory than bf16."""
        torch.cuda.reset_peak_memory_stats()

        # BF16 baseline
        model_bf16 = nn.Sequential(
            nn.Linear(1024, 2048, bias=False, dtype=torch.bfloat16, device="cuda"),
            nn.ReLU(),
            nn.Linear(2048, 1024, bias=False, dtype=torch.bfloat16, device="cuda"),
        )
        x = torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda")
        out = model_bf16(x)
        out.sum().backward()
        bf16_mem = torch.cuda.max_memory_allocated()
        del model_bf16, x, out
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # FP8
        from lset.distributed.fp8 import apply_fp8_training
        model_fp8 = nn.Sequential(
            nn.Linear(1024, 2048, bias=False, dtype=torch.bfloat16, device="cuda"),
            nn.ReLU(),
            nn.Linear(2048, 1024, bias=False, dtype=torch.bfloat16, device="cuda"),
        )
        apply_fp8_training(model_fp8, recipe="rowwise")
        x = torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda")
        out = model_fp8(x)
        out.sum().backward()
        fp8_mem = torch.cuda.max_memory_allocated()

        # FP8 should not use drastically more memory (allow 2x margin for scales etc)
        assert fp8_mem < bf16_mem * 2.0, (
            f"FP8 mem {fp8_mem/1e6:.1f}MB >> BF16 mem {bf16_mem/1e6:.1f}MB"
        )


def _is_distributed():
    return os.environ.get("RANK") is not None


requires_distributed = pytest.mark.skipif(
    not _is_distributed(),
    reason="Requires torchrun with 2+ GPUs",
)


@requires_distributed
def test_fp8_fsdp2_distributed():
    """FP8 + FSDP2 training on 2 GPUs."""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    from lset.models import get_model_spec
    from lset.distributed.parallel import build_parallel_model, ParallelConfig
    from lset.distributed.fp8 import apply_fp8_training

    model_path = os.path.expanduser("~/models/Qwen3-Embedding-0.6B")
    spec = get_model_spec("qwen3")
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)

    # FP8 → TP → FSDP2
    apply_fp8_training(model, recipe="rowwise")
    pconfig = ParallelConfig(dp_size=2, tp_size=1)
    model, mesh = build_parallel_model(model, config, pconfig)

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    losses = []
    for step in range(10):
        x = torch.randint(0, 1000, (2, 32), device="cuda")
        mask = torch.ones(2, 32, device="cuda", dtype=torch.long)
        out = model(x, mask)
        loss = out["hidden_states"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN in losses"
    if rank == 0:
        print(f"FP8+FSDP2 losses: {losses[0]:.2f} → {losses[-1]:.2f}")

    dist.destroy_process_group()
