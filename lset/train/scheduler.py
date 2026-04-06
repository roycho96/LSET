"""Learning rate schedulers with warmup support."""

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, scheduler_type: str, max_steps: int, warmup_steps: int = 0, **kwargs):
    """Build a learning rate scheduler with optional warmup.

    Args:
        optimizer: The optimizer.
        scheduler_type: One of "cosine", "linear", "wsd", "constant".
        max_steps: Total training steps.
        warmup_steps: Number of warmup steps.
        **kwargs: Extra args (e.g., stable_ratio for wsd).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Post-warmup progress (0 to 1)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        if scheduler_type == "constant":
            return 1.0
        elif scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            return 1.0 - progress
        elif scheduler_type == "wsd":
            # Warmup-Stable-Decay
            stable_ratio = kwargs.get("stable_ratio", 0.7)
            decay_ratio = 1.0 - stable_ratio
            if progress < stable_ratio:
                return 1.0
            else:
                decay_progress = (progress - stable_ratio) / max(1e-8, decay_ratio)
                return 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)
