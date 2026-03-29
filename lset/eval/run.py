"""MTEB evaluation entry point — called by lset CLI."""

from __future__ import annotations

from pathlib import Path

from lset.config import LSETConfig


def run_eval(config: LSETConfig):
    """Run MTEB evaluation with the given config."""
    import mteb
    from lset.eval import LSETMTEBModel

    model_path = str(Path(config.model.path).expanduser())
    pooling = config.model.pooling if config.model.pooling != "auto" else None

    model = LSETMTEBModel.from_pretrained(
        model_path,
        pooling=pooling,
        max_length=config.eval.max_length,
    )

    task_names = config.eval.tasks
    if not task_names:
        task_names = ["STSBenchmark"]

    tasks = mteb.get_tasks(tasks=task_names)
    results = mteb.evaluate(
        model, tasks,
        encode_kwargs={"batch_size": config.eval.batch_size},
    )

    # Print summary
    print(f"\n{'='*60}")
    print("MTEB Evaluation Results")
    print(f"{'='*60}")
    for tr in results:
        for split in ("test", "dev", "validation"):
            if split in tr.scores and tr.scores[split]:
                score = tr.scores[split][0].get("main_score", 0)
                print(f"  {tr.task_name:30s}  {split:10s}  {score:.4f}")
                break
    print(f"{'='*60}")
