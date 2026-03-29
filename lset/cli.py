"""LSET CLI entrypoint.

Usage:
    lset train --config configs/qwen3_embed_quick.yaml [overrides...]
    lset eval  --config configs/qwen3_embed_quick.yaml [overrides...]
"""

from __future__ import annotations

import sys


HELP_TEXT = """\
LSET - Large Scale Embedding Trainer

Commands:
  train       Train an embedding model
  eval        Evaluate with MTEB

Usage:
  lset train --config configs/qwen3_embed_quick.yaml [--section.field value ...]
  lset eval  --config configs/qwen3_embed_quick.yaml [--eval.tasks STS12,STS13 ...]

Examples:
  lset train --config configs/qwen3_embed_quick.yaml --training.max_steps 100
  lset train --config configs/qwen3_embed_packed.yaml --training.batch_size 16
  lset eval  --config configs/qwen3_embed_quick.yaml --eval.tasks STSBenchmark
"""


def _extract_config_path(args: list[str]) -> str | None:
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            return args[i + 1]
    return None


def cmd_train(args: list[str]):
    from lset.config import LSETConfig, parse_overrides

    config_path = _extract_config_path(args)
    if not config_path:
        print("Error: --config is required\n")
        print("Usage: lset train --config <config.yaml> [overrides...]")
        sys.exit(1)

    config = LSETConfig.from_yaml(config_path)
    overrides = parse_overrides(args)
    config.apply_overrides(overrides)
    config.validate()

    world_size = config.distributed.dp_size * config.distributed.tp_size

    if world_size > 1:
        import subprocess
        # Rebuild override args for subprocess
        override_args = []
        for k, v in overrides:
            override_args.extend([f"--{k}", v])
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "-m", "lset.train.main",
            "--config", config_path,
        ] + override_args
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        from lset.train.main import train
        train(config)


def cmd_eval(args: list[str]):
    from lset.config import LSETConfig, parse_overrides

    config_path = _extract_config_path(args)
    if not config_path:
        print("Error: --config is required\n")
        print("Usage: lset eval --config <config.yaml> [overrides...]")
        sys.exit(1)

    config = LSETConfig.from_yaml(config_path)
    overrides = parse_overrides(args)
    config.apply_overrides(overrides)

    from lset.eval.run import run_eval
    run_eval(config)


def main():
    if len(sys.argv) < 2:
        print(HELP_TEXT)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "train":
        cmd_train(args)
    elif command == "eval":
        cmd_eval(args)
    elif command in ("-h", "--help", "help"):
        print(HELP_TEXT)
    else:
        print(f"Unknown command: {command}\n")
        print(HELP_TEXT)
        sys.exit(1)


if __name__ == "__main__":
    main()
