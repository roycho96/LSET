#!/usr/bin/env python
"""Entry point for Qwen3 embedding training.

Usage:
    # Single GPU
    python scripts/train_qwen3_embed.py --model_path ~/models/Qwen3-Embedding-0.6B

    # 2 GPU FSDP2
    torchrun --nproc_per_node=2 scripts/train_qwen3_embed.py \
        --model_path ~/models/Qwen3-Embedding-0.6B --dp_size 2

    # With packing + GradCache
    python scripts/train_qwen3_embed.py --model_path ~/models/Qwen3-Embedding-0.6B \
        --packed --grad_cache --gc_chunk_size 4

    # With real data
    python scripts/train_qwen3_embed.py --model_path ~/models/Qwen3-Embedding-0.6B \
        --data_path data/train.jsonl --gradient_accumulation_steps 4 \
        --scheduler cosine --warmup_steps 100
"""

import argparse
import json
import tempfile
from pathlib import Path

from lset.tokenization.loader import load_tokenizer
from lset.tokenization.templates import QWEN3_EMBEDDING_TEMPLATE
from lset.data.dataset import EmbeddingDataset
from lset.train.engine import TrainingEngine


def create_synthetic_data(path: Path, num_samples: int = 100):
    """Create a small synthetic dataset for testing."""
    samples = []
    for i in range(num_samples):
        samples.append({
            "query": f"What is topic number {i}?",
            "positive": f"Topic {i} is about the study of concept {i} in the field of science.",
            "negatives": [f"This is an unrelated document about item {(i + 37) % num_samples}."],
        })
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 embedding model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to JSONL training data. If not provided, uses synthetic data.")
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--packed", action="store_true", help="Use sequence packing")
    parser.add_argument("--grad_cache", action="store_true", help="Use GradCache")
    parser.add_argument("--gc_chunk_size", type=int, default=16, help="GradCache chunk size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save checkpoint every N optimizer steps (0=disabled)")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "wsd", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="lset")
    parser.add_argument("--num_hard_negatives", type=int, default=0,
                        help="Number of hard negatives to use per query (0=all)")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)

    # Data
    if args.data_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        data_path = create_synthetic_data(Path(tmp.name))
    else:
        data_path = Path(args.data_path)

    dataset = EmbeddingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        template=QWEN3_EMBEDDING_TEMPLATE,
        num_hard_negatives=args.num_hard_negatives,
    )

    engine = TrainingEngine(
        model_name="qwen3",
        model_path=args.model_path,
        dataset=dataset,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        temperature=args.temperature,
        log_interval=args.log_interval,
        packed=args.packed,
        use_grad_cache=args.grad_cache,
        gc_chunk_size=args.gc_chunk_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    engine.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
