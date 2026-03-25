"""Training engine for embedding models."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lset.models import get_model_spec
from lset.tasks.bi_encoder import BiEncoderTask
from lset.data.collator import LeftPadCollator
from lset.distributed.parallel import setup_fsdp2


class TrainingEngine:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        dataset,
        dp_size: int = 1,
        batch_size: int = 8,
        lr: float = 1e-5,
        max_steps: int = 1000,
        grad_clip: float = 1.0,
        temperature: float = 0.02,
        matryoshka_dims: list[int] | None = None,
        log_interval: int = 10,
    ):
        self.dp_size = dp_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if dp_size > 1:
            import torch.distributed as dist
            dist.init_process_group("nccl")
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

        # FSDP2
        if dp_size > 1:
            self.model, self.mesh = setup_fsdp2(self.model, dp_size)

        # Task
        self.task = BiEncoderTask(
            pooling=spec.default_pooling,
            temperature=temperature,
            matryoshka_dims=matryoshka_dims,
        )

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr, fused=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps)

        # Dataloader
        pad_token_id = config.vocab_size - 1  # fallback
        # Try to use eos_token_id from HF config
        import json
        with open(f"{model_path}/config.json") as f:
            hf_config = json.load(f)
        pad_token_id = hf_config.get("eos_token_id", pad_token_id)

        collator = LeftPadCollator(pad_token_id=pad_token_id)

        sampler = None
        if dp_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=dp_size, rank=self.rank)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            collate_fn=collator,
            sampler=sampler,
            drop_last=True,
        )
        self.device = device

    def train(self):
        self.model.train()
        step = 0
        while step < self.max_steps:
            for batch in self.dataloader:
                if step >= self.max_steps:
                    break

                query_batch = {k: v.to(self.device) for k, v in batch["query"].items()}
                doc_batch = {k: v.to(self.device) for k, v in batch["doc"].items()}
                neg_batch = None
                if "neg" in batch:
                    neg_batch = {k: v.to(self.device) for k, v in batch["neg"].items()}

                out = self.task(self.model, query_batch, doc_batch, neg_batch)
                loss = out["loss"]
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.rank == 0 and step % self.log_interval == 0:
                    print(f"step={step} loss={loss.item():.4f}")

                step += 1

        if self.dp_size > 1:
            import torch.distributed as dist
            dist.destroy_process_group()
