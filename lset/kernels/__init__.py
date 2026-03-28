"""Triton kernels for embedding training optimizations."""

from .fused_normalize import normalize, fused_l2_normalize
from .fused_loss import fused_dense_loss, should_use_fused
from .fused_rmsnorm import rms_norm, fused_rms_norm
from .fused_rope import apply_rotary_pos_emb, fused_apply_rotary_pos_emb
from .fused_swiglu import swiglu, fused_swiglu

__all__ = [
    "normalize", "fused_l2_normalize",
    "fused_dense_loss", "should_use_fused",
    "rms_norm", "fused_rms_norm",
    "apply_rotary_pos_emb", "fused_apply_rotary_pos_emb",
    "swiglu", "fused_swiglu",
]
