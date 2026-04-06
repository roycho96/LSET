"""Checkpoint save/load using PyTorch Distributed Checkpoint (DCP)."""

import os

import torch
import torch.distributed as dist


def save_checkpoint(model, optimizer, step: int, output_dir: str):
    """Save FSDP2-compatible checkpoint using DCP."""
    checkpoint_dir = os.path.join(output_dir, f"step_{step}")

    if dist.is_initialized():
        import torch.distributed.checkpoint as dcp

        state = {"model": model, "optimizer": optimizer}
        dcp.save(state, checkpoint_id=checkpoint_dir)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            os.path.join(checkpoint_dir, "checkpoint.pt"),
        )

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # Save step metadata
        meta_path = os.path.join(checkpoint_dir, "meta.pt")
        torch.save({"step": step}, meta_path)


def load_checkpoint(model, optimizer, checkpoint_dir: str) -> int:
    """Load checkpoint. Returns the step number."""
    if dist.is_initialized():
        import torch.distributed.checkpoint as dcp

        state = {"model": model, "optimizer": optimizer}
        dcp.load(state, checkpoint_id=checkpoint_dir)
    else:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    meta_path = os.path.join(checkpoint_dir, "meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=True)
        return meta["step"]
    return 0
