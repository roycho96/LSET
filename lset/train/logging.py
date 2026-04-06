"""Training logger with console and optional wandb support."""

import time


class TrainLogger:
    """Simple training logger supporting console + optional wandb."""

    def __init__(
        self,
        use_wandb: bool = False,
        project: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ):
        self.use_wandb = use_wandb
        self._step_start_time = time.time()
        self._total_samples = 0

        if use_wandb:
            import wandb

            wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict, step: int):
        """Log metrics at a given step."""
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" ".join(parts))

        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

    def update_throughput(self, num_samples: int) -> dict:
        """Compute throughput metrics since last call."""
        now = time.time()
        elapsed = now - self._step_start_time
        self._step_start_time = now
        if elapsed > 0:
            return {"samples_per_sec": num_samples / elapsed}
        return {}

    def finish(self):
        if self.use_wandb:
            import wandb

            wandb.finish()
