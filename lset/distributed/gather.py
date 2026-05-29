"""Cross-GPU gather with gradient passthrough."""

import torch
import torch.distributed as dist


class _GatherWithGrad(torch.autograd.Function):
    """All-gather that passes gradients back to the local shard."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        # Make our own shard's entry the original tensor (preserves grad_fn)
        gathered[rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Slice out only this rank's gradients
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        return grad_output[start:end].contiguous()


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather embeddings across GPUs, preserving gradients."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return _GatherWithGrad.apply(tensor)
