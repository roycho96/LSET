#!/usr/bin/env python
"""Measure memory usage with and without SequenceParallel.

Usage:
    torchrun --nproc_per_node=2 scripts/measure_tp_memory.py \
        --model_path ~/models/Qwen3-Embedding-0.6B --mode sp
    torchrun --nproc_per_node=2 scripts/measure_tp_memory.py \
        --model_path ~/models/Qwen3-Embedding-0.6B --mode no_sp
"""

import argparse
import os
import torch
import torch.distributed as dist

from lset.models import get_model_spec
from lset.distributed.parallel import build_parallel_model, ParallelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", choices=["sp", "no_sp"], required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    spec = get_model_spec("qwen3")
    config = spec.config_cls.from_hf_json(f"{args.model_path}/config.json")
    model = spec.model_cls(config)
    state_dict = spec.weight_converter(args.model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)

    use_sp = args.mode == "sp"
    pconfig = ParallelConfig(dp_size=1, tp_size=2, use_sequence_parallel=use_sp)
    model, mesh = build_parallel_model(model, config, pconfig)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Run a few forward-backward to warm up
    for i in range(5):
        x = torch.randint(0, 1000, (args.batch_size, args.seq_len), device="cuda")
        mask = torch.ones(args.batch_size, args.seq_len, device="cuda", dtype=torch.long)
        out = model(x, mask)
        loss = out["hidden_states"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    if rank == 0:
        print(f"Mode: {args.mode} | SP: {use_sp}")
        print(f"Batch: {args.batch_size} x {args.seq_len}")
        print(f"Peak GPU memory: {peak_mb:.1f} MB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
