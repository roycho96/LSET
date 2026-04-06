"""Training engine for embedding models."""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from lset.models import get_model_spec
from lset.tasks.bi_encoder import BiEncoderTask
from lset.tasks.grad_cache import GradCacheWrapper
from lset.train.data.collator import LeftPadCollator, RightPadCollator, EmbeddingCollator
from lset.distributed.parallel import setup_fsdp2, build_parallel_model, ParallelConfig
from lset.train.scheduler import build_scheduler
from lset.train.checkpoint import save_checkpoint, load_checkpoint
from lset.train.logging import TrainLogger


def _clip_grad_norm_tp(model: nn.Module, max_norm: float):
    """Clip gradients for TP models, avoiding mixed-mesh DTensor issues."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # Convert to plain float to avoid DTensor mesh mismatch
            param_norm = p.grad.detach().float().norm(2.0).item()
            total_norm_sq += param_norm ** 2
    total_norm = total_norm_sq ** 0.5
    clip_coef = max_norm / max(total_norm, 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)


class TrainingEngine:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        dataset,
        dp_size: int = 1,
        tp_size: int = 1,
        batch_size: int = 8,
        lr: float = 1e-5,
        max_steps: int = 1000,
        grad_clip: float = 1.0,
        temperature: float = 0.02,
        matryoshka_dims: list[int] | None = None,
        log_interval: int = 10,
        packed: bool = False,
        use_grad_cache: bool = False,
        gc_chunk_size: int = 16,
        gc_token_budget: int | None = None,
        gc_selective_keep: float = 1.0,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 0,
        output_dir: str = "./output",
        resume_from: str | None = None,
        scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        use_wandb: bool = False,
        wandb_project: str = "lset",
        use_label_matrix: bool = False,
        collator=None,
        # LoRA/QLoRA
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_targets: list[str] | None = None,
        qlora: bool = False,
        qlora_block_size: int = 64,
        # FP8 training
        fp8: bool = False,
        fp8_recipe: str = "rowwise",
        # Attention backend
        attn_backend: str = "auto",
        # CUDA graph (padded mode only)
        cuda_graph: bool = False,
        # Truncated InfoNCE
        top_k: int | None = None,
        # Cascade InfoNCE
        cascade: bool = False,
        cascade_d_small: int = 64,
        cascade_K_prime: int = 256,
    ):
        # Validate CUDA graph compatibility
        if cuda_graph:
            from lset.train.cuda_graph import validate_cuda_graph_config
            validate_cuda_graph_config(packed, use_grad_cache, compile_model=False)

        # Validate incompatible options
        if fp8 and (lora or qlora):
            raise ValueError(
                "FP8 training + LoRA is not supported (torchtune#2833). "
                "Use FP8 for full fine-tuning or LoRA/QLoRA without FP8."
            )
        if qlora and tp_size > 1:
            raise ValueError(
                "QLoRA + Tensor Parallelism is not supported (NF4 + TP not "
                "implemented in torchao). Use LoRA + TP or QLoRA without TP."
            )

        # Set attention backend before model construction
        from lset.models.decoder.qwen3.attention import set_attn_backend
        set_attn_backend(attn_backend)

        self.dp_size = dp_size
        self.tp_size = tp_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.use_grad_cache = use_grad_cache
        self.grad_accum_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.resume_from = resume_from
        self.use_label_matrix = use_label_matrix
        self.needs_dist_cleanup = False
        self.use_lora = lora or qlora
        self.use_fp8 = fp8
        self.use_cuda_graph = cuda_graph

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        needs_dist = dp_size > 1 or tp_size > 1
        if needs_dist:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group("nccl")
                self.needs_dist_cleanup = True
            torch.cuda.set_device(self.local_rank)

        device = torch.device(f"cuda:{self.local_rank}")

        # Build model
        spec = get_model_spec(model_name)
        config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
        self.model = spec.model_cls(config)

        # Load weights
        state_dict = spec.weight_converter(model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device=device, dtype=torch.bfloat16)

        # Apply FP8 (before TP/FSDP, after weights loaded)
        if fp8:
            from lset.train.quantization.fp8 import apply_fp8_training
            apply_fp8_training(self.model, recipe=fp8_recipe)

        # Apply QLoRA (quantize then LoRA — before TP/FSDP)
        lora_target_list = lora_targets if lora_targets else None
        if qlora:
            from lset.train.lora import apply_qlora
            apply_qlora(
                self.model, r=lora_r, alpha=lora_alpha,
                target_modules=lora_target_list or ("q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"),
                dropout=lora_dropout, block_size=qlora_block_size,
            )
        elif lora:
            from lset.train.lora import apply_lora
            apply_lora(
                self.model, r=lora_r, alpha=lora_alpha,
                target_modules=lora_target_list or ("q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"),
                dropout=lora_dropout,
            )

        # Apply parallelism
        if tp_size > 1:
            # 2D parallelism (TP + optional FSDP)
            # Enable SequenceParallel for padded mode (not packed)
            pconfig = ParallelConfig(
                dp_size=dp_size, tp_size=tp_size, mp_dtype=torch.bfloat16,
                use_sequence_parallel=not packed,
                use_lora=self.use_lora,
            )
            self.model, self.mesh = build_parallel_model(
                self.model, config, pconfig,
            )
        elif dp_size > 1:
            # FSDP2 only (legacy path)
            self.model, self.mesh = setup_fsdp2(self.model, dp_size)

        # Task
        self.task = BiEncoderTask(
            pooling=spec.default_pooling,
            temperature=temperature,
            matryoshka_dims=matryoshka_dims,
            top_k=top_k,
            cascade=cascade,
            cascade_d_small=cascade_d_small,
            cascade_K_prime=cascade_K_prime,
        )

        # GradCache
        self.grad_cache = None
        if use_grad_cache:
            self.grad_cache = GradCacheWrapper(
                self.task, chunk_size=gc_chunk_size,
                token_budget=gc_token_budget,
                selective_keep=gc_selective_keep,
            )

        # Optimizer — only LoRA params when using LoRA/QLoRA
        if self.use_lora:
            from lset.train.lora import get_lora_params
            opt_params = get_lora_params(self.model)
        else:
            opt_params = list(self.model.parameters())
        self.optimizer = AdamW(opt_params, lr=lr, fused=True)

        # Scheduler
        self.scheduler = build_scheduler(
            self.optimizer, scheduler_type, max_steps, warmup_steps,
        )

        # Logger
        self.logger = TrainLogger(
            use_wandb=use_wandb,
            project=wandb_project,
            config={"model": model_name, "batch_size": batch_size, "lr": lr},
        )

        # Collator
        is_new_format = hasattr(dataset, 'format')
        if collator is not None:
            actual_collator = collator
        elif is_new_format:
            from lset.tokenization import load_tokenizer
            tokenizer = load_tokenizer(model_path)
            actual_collator = EmbeddingCollator(
                tokenizer=tokenizer,
                max_length=dataset.max_length,
                packed=packed,
            )
            self.use_label_matrix = True
        elif packed:
            from lset.train.data.packed_collator import PackedCollator
            actual_collator = PackedCollator()
        else:
            with open(f"{model_path}/config.json") as f:
                hf_config = json.load(f)
            pad_token_id = hf_config.get("pad_token_id") or hf_config.get("eos_token_id", config.vocab_size - 1)
            if spec.default_padding_side == "right":
                actual_collator = RightPadCollator(pad_token_id=pad_token_id)
            else:
                actual_collator = LeftPadCollator(pad_token_id=pad_token_id)

        sampler = None
        if dp_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=dp_size, rank=self.rank)

        # When using TP without DP, all TP ranks must see the same data.
        # Use a fixed-seed generator so shuffle produces identical order on all ranks.
        shuffle = sampler is None
        generator = None
        if shuffle and tp_size > 1:
            generator = torch.Generator()
            generator.manual_seed(42)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=actual_collator,
            sampler=sampler,
            drop_last=True,
            generator=generator,
        )
        self.device = device

        # CUDA graph wrapper (set up lazily on first batch to know seq_length)
        self.cuda_graph_wrapper = None
        if self.use_cuda_graph:
            self._cuda_graph_seq_length = None  # set from first batch

    def _to_device(self, d: dict) -> dict:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in d.items()}

    def train(self):
        self.model.train()
        start_step = 0

        # Resume from checkpoint
        if self.resume_from is not None:
            start_step = load_checkpoint(self.model, self.optimizer, self.resume_from)
            if self.rank == 0:
                print(f"Resumed from step {start_step}")

        step = start_step
        self.optimizer.zero_grad()

        while step < self.max_steps:
            for batch in self.dataloader:
                if step >= self.max_steps:
                    break

                query_batch = self._to_device(batch["query"])
                doc_batch = self._to_device(batch["doc"])
                labels = batch.get("labels")
                scores = batch.get("scores")

                if self.use_grad_cache:
                    loss = self.grad_cache(self.model, query_batch, doc_batch,
                                           labels=labels, scores=scores)
                else:
                    neg_batch = None
                    if "neg" in batch:
                        neg_batch = self._to_device(batch["neg"])
                    out = self.task(self.model, query_batch, doc_batch, neg_batch,
                                    labels=labels, scores=scores)
                    loss = out["loss"]
                    scaled_loss = loss / self.grad_accum_steps
                    scaled_loss.backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    if self.tp_size > 1:
                        _clip_grad_norm_tp(self.model, self.grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                if self.rank == 0 and step % self.log_interval == 0:
                    metrics = {
                        "loss": loss_val,
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    self.logger.log(metrics, step)

                # Checkpoint
                if (self.save_steps > 0 and (step + 1) % self.save_steps == 0):
                    save_checkpoint(self.model, self.optimizer, step + 1, self.output_dir)
                    if self.rank == 0:
                        print(f"Saved checkpoint at step {step + 1}")

                step += 1

        if self.needs_dist_cleanup:
            import torch.distributed as dist
            dist.destroy_process_group()

        self.logger.finish()
