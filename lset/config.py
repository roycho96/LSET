"""YAML config system for LSET."""

from __future__ import annotations

import typing

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import List
from typing import Optional
from typing import get_type_hints


def _resolve_path(p: str) -> str:
    """Expand ~ and resolve relative paths."""
    if not p:
        return p
    return str(Path(p).expanduser())


# ─── Config Sections ──────────────────────────────────────────────────


@dataclass
class ModelConfig:
    path: str = ""
    pooling: str = "auto"  # auto | last_token | mean | cls
    padding_side: str = "auto"  # auto | left | right
    fused_projections: bool = False  # fuse QKV and GateUp


@dataclass
class DataConfig:
    train_path: str = ""
    max_seq_length: int = 128
    num_hard_negatives: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 2e-5
    max_steps: Optional[int] = None  # None = use epochs
    epochs: int = 1
    scheduler: str = "cosine"  # cosine | linear | wsd | constant
    warmup_steps: int = 0
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    temperature: float = 0.02
    top_k: Optional[int] = None  # truncated InfoNCE: keep only top-K negatives
    cascade: bool = False  # cascade proxy InfoNCE (Phase 11)
    cascade_d_small: int = 64
    cascade_K_prime: int = 256
    matryoshka_dims: Optional[List[int]] = None


@dataclass
class PackingConfig:
    enabled: bool = False


@dataclass
class GradCacheConfig:
    enabled: bool = False
    token_budget: int = 4096  # primary: max tokens per chunk
    chunk_size: Optional[int] = None  # fallback: fixed sequence count per chunk
    selective_backward_keep: float = 1.0  # fraction of samples to re-encode in backward (1.0=all)


@dataclass
class CompileConfig:
    enabled: bool = False
    dynamic: bool = True
    backend: str = "inductor"
    mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"


@dataclass
class ActivationCheckpointConfig:
    """Activation checkpointing on transformer blocks."""

    enabled: bool = False
    mode: str = "selective"  # "selective" (op-level SAC) | "full"
    ratio: float = 1.0  # fraction of layers (from the bottom) to checkpoint


@dataclass
class CudaGraphConfig:
    enabled: bool = False
    token_budget: Optional[int] = None


@dataclass
class KernelsConfig:
    fused: bool = True  # master switch
    fused_residual_rmsnorm: bool = True
    fused_pool_normalize: bool = True
    fused_layernorm: bool = False  # disabled: PyTorch cuDNN is faster


@dataclass
class AttentionConfig:
    backend: str = "auto"  # auto | flash_attn | varlen_attn | sdpa


@dataclass
class DistributedConfig:
    dp_size: int = 1
    tp_size: int = 1
    async_tp: bool = False  # requires compile.enabled=True


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: Optional[List[str]] = None


@dataclass
class QloraConfig:
    enabled: bool = False
    block_size: int = 64


@dataclass
class Fp8Config:
    enabled: bool = False
    recipe: str = "rowwise"  # rowwise | tensorwise


@dataclass
class CheckpointConfig:
    save_steps: int = 0
    output_dir: str = "output/"
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    log_interval: int = 10
    wandb: bool = False
    wandb_project: str = "lset"


@dataclass
class EvalConfig:
    tasks: Optional[List[str]] = None
    batch_size: int = 32
    max_length: int = 512


# ─── Top-Level Config ─────────────────────────────────────────────────


@dataclass
class LSETConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    grad_cache: GradCacheConfig = field(default_factory=GradCacheConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    cuda_graph: CudaGraphConfig = field(default_factory=CudaGraphConfig)
    kernels: KernelsConfig = field(default_factory=KernelsConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    qlora: QloraConfig = field(default_factory=QloraConfig)
    fp8: Fp8Config = field(default_factory=Fp8Config)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ── Constructors ──

    @classmethod
    def from_yaml(cls, path: str | Path) -> LSETConfig:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> LSETConfig:
        """Build config from a (possibly partial) dict."""
        cfg = cls()
        section_map = {f.name: f for f in fields(cls)}
        for section_name, section_data in d.items():
            if section_name not in section_map:
                continue
            section_obj = getattr(cfg, section_name)
            if isinstance(section_data, dict):
                hints = get_type_hints(type(section_obj))
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        # Coerce YAML values to match the dataclass field type
                        target_type = hints.get(key, type(value))
                        if isinstance(value, str) and target_type in (int, float):
                            value = target_type(value)
                        setattr(section_obj, key, value)
        return cfg

    # ── CLI Override ──

    def apply_overrides(self, overrides: list[tuple[str, str]]):
        """Apply CLI overrides in dotted notation."""
        for key, raw_value in overrides:
            parts = key.split(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Override key must be section.field: {key}")
            section_name, field_name = parts

            if not hasattr(self, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(self, section_name)
            if not hasattr(section, field_name):
                raise ValueError(f"Unknown field {field_name} in section {section_name}")

            # Get target type from dataclass field annotations
            hints = get_type_hints(type(section))
            target_type = hints.get(field_name, str)
            coerced = _coerce_value(raw_value, target_type)
            setattr(section, field_name, coerced)

    # ── Validation ──

    def validate(self):
        """Validate config. Raise clear errors for invalid combinations."""
        if self.cuda_graph.enabled and self.packing.enabled and not self.cuda_graph.token_budget:
            raise ValueError("cuda_graph.enabled + packing.enabled requires cuda_graph.token_budget")
        if self.qlora.enabled and self.distributed.tp_size > 1:
            raise ValueError("QLoRA + TP is not supported (NF4 + TP not in torchao)")
        if self.fp8.enabled and (self.lora.enabled or self.qlora.enabled):
            raise ValueError("FP8 + LoRA/QLoRA is not supported")
        if self.cuda_graph.enabled and self.compile.enabled:
            raise ValueError("CUDA graph + torch.compile is redundant (use one or the other)")
        if self.cuda_graph.enabled and self.grad_cache.enabled:
            raise ValueError("CUDA graph + GradCache is not supported")

    # ── Serialization ──

    def to_yaml(self, path: str | Path):
        """Save config to YAML for reproducibility."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to plain dict."""
        return asdict(self)


# ─── Override Parsing ──────────────────────────────────────────────────


def parse_overrides(args: list[str]) -> list[tuple[str, str]]:
    """Parse CLI overrides from argument list."""
    overrides = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and "." in arg and arg != "--config":
            key = arg[2:]  # strip --
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                value = "true"  # bare flag
                i += 1
            overrides.append((key, value))
        else:
            i += 1
    return overrides


def _coerce_value(raw: str, target_type) -> object:
    """Coerce a string value to the appropriate Python type."""
    origin = getattr(target_type, "__origin__", None)

    # Handle Optional[X] → extract X
    if origin is typing.Union:
        type_args = target_type.__args__
        non_none = [t for t in type_args if t is not type(None)]
        if raw.lower() in ("null", "none", ""):
            return None
        if non_none:
            return _coerce_value(raw, non_none[0])

    # Handle List[X]
    if origin is list:
        inner = target_type.__args__[0] if target_type.__args__ else str
        if raw.lower() in ("null", "none", ""):
            return None
        items = [s.strip() for s in raw.split(",")]
        return [_coerce_value(item, inner) for item in items]

    # Primitives
    if target_type is bool:
        return raw.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw
