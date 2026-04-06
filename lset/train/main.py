"""Main training entry point — called by lset CLI or directly."""

from __future__ import annotations

import os
import sys

from lset.config import LSETConfig
from lset.config import parse_overrides


def train(config: LSETConfig):
    """Run training with the given config."""
    from pathlib import Path

    import torch

    from lset.models.registry import detect_model_type
    from lset.tokenization import load_tokenizer
    from lset.train.data.dataset import EmbeddingDataset
    from lset.train.engine import TrainingEngine

    model_path = str(Path(config.model.path).expanduser())
    model_name = detect_model_type(model_path)

    # Set kernel env vars before importing model/kernels
    if not config.kernels.fused:
        os.environ["LSET_DISABLE_FUSED_RESIDUAL_RMSNORM"] = "1"
        os.environ["LSET_DISABLE_FUSED_POOL_NORMALIZE"] = "1"
        os.environ["LSET_DISABLE_FUSED_LAYERNORM"] = "1"
    else:
        if not config.kernels.fused_residual_rmsnorm:
            os.environ["LSET_DISABLE_FUSED_RESIDUAL_RMSNORM"] = "1"
        if not config.kernels.fused_pool_normalize:
            os.environ["LSET_DISABLE_FUSED_POOL_NORMALIZE"] = "1"
        if not config.kernels.fused_layernorm:
            os.environ["LSET_DISABLE_FUSED_LAYERNORM"] = "1"

    # Set attention backend
    from lset.models.decoder.qwen3.attention import set_attn_backend

    set_attn_backend(config.attention.backend)

    # Seed
    torch.manual_seed(config.training.seed)

    # Dataset
    data_path = str(Path(config.data.train_path).expanduser())
    tokenizer = load_tokenizer(model_path)
    dataset = EmbeddingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.data.max_seq_length,
        num_hard_negatives=config.data.num_hard_negatives or 0,
    )

    # Compute max_steps from epochs if needed
    max_steps = config.training.max_steps
    if max_steps is None:
        steps_per_epoch = len(dataset) // config.training.batch_size
        max_steps = steps_per_epoch * config.training.epochs

    # Build engine
    engine = TrainingEngine(
        model_name=model_name,
        model_path=model_path,
        dataset=dataset,
        dp_size=config.distributed.dp_size,
        tp_size=config.distributed.tp_size,
        batch_size=config.training.batch_size,
        lr=config.training.lr,
        max_steps=max_steps,
        grad_clip=config.training.grad_clip,
        temperature=config.training.temperature,
        matryoshka_dims=config.training.matryoshka_dims,
        log_interval=config.logging.log_interval,
        packed=config.packing.enabled,
        use_grad_cache=config.grad_cache.enabled,
        gc_chunk_size=config.grad_cache.chunk_size or 16,
        gc_token_budget=config.grad_cache.token_budget if config.grad_cache.enabled else None,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_steps=config.checkpoint.save_steps,
        output_dir=config.checkpoint.output_dir,
        resume_from=config.checkpoint.resume_from,
        scheduler_type=config.training.scheduler,
        warmup_steps=config.training.warmup_steps,
        use_wandb=config.logging.wandb,
        wandb_project=config.logging.wandb_project,
        lora=config.lora.enabled,
        lora_r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        lora_targets=config.lora.targets,
        qlora=config.qlora.enabled,
        qlora_block_size=config.qlora.block_size,
        fp8=config.fp8.enabled,
        fp8_recipe=config.fp8.recipe,
        attn_backend=config.attention.backend,
        cuda_graph=config.cuda_graph.enabled,
        top_k=config.training.top_k,
        cascade=config.training.cascade,
        cascade_d_small=config.training.cascade_d_small,
        cascade_K_prime=config.training.cascade_K_prime,
        gc_selective_keep=config.grad_cache.selective_backward_keep,
    )

    # Compile
    if config.compile.enabled:
        import torch

        engine.model = torch.compile(
            engine.model,
            dynamic=config.compile.dynamic,
            backend=config.compile.backend,
        )

    engine.train()

    # Save final config for reproducibility
    from pathlib import Path

    out_dir = Path(config.checkpoint.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")


if __name__ == "__main__":
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if not config_path:
        print("Usage: python -m lset.train.main --config <config.yaml> [overrides...]")
        sys.exit(1)

    config = LSETConfig.from_yaml(config_path)
    overrides = parse_overrides(sys.argv[1:])
    config.apply_overrides(overrides)
    config.validate()
    train(config)
