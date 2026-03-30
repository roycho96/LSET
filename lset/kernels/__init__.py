"""Triton kernels for embedding training optimizations."""

from .fused_normalize import normalize, fused_l2_normalize
from .fused_loss import fused_dense_loss, should_use_fused
from .fused_rmsnorm import rms_norm, fused_rms_norm
from .fused_residual_rmsnorm import residual_rms_norm, fused_residual_rms_norm
from .fused_layernorm import layer_norm, fused_layer_norm
from .fused_residual_layernorm import residual_layer_norm, fused_residual_layer_norm
from .fused_pool_normalize import fused_pool_normalize
from .fused_rope import apply_rotary_pos_emb, fused_apply_rotary_pos_emb
from .fused_swiglu import swiglu, fused_swiglu
from .fused_geglu import geglu, fused_geglu
from .triton_segment_pool import triton_segment_mean_pool

__all__ = [
    "normalize", "fused_l2_normalize",
    "fused_dense_loss", "should_use_fused",
    "rms_norm", "fused_rms_norm",
    "residual_rms_norm", "fused_residual_rms_norm",
    "layer_norm", "fused_layer_norm",
    "residual_layer_norm", "fused_residual_layer_norm",
    "fused_pool_normalize",
    "apply_rotary_pos_emb", "fused_apply_rotary_pos_emb",
    "swiglu", "fused_swiglu",
    "geglu", "fused_geglu",
]
