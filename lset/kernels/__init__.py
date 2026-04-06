"""Triton kernels for embedding training optimizations."""

from lset.kernels.normalize import normalize, fused_l2_normalize
from lset.kernels.loss import fused_dense_loss, should_use_fused
from lset.kernels.rmsnorm import rms_norm, fused_rms_norm
from lset.kernels.residual_rmsnorm import residual_rms_norm, fused_residual_rms_norm
from lset.kernels.layernorm import layer_norm, fused_layer_norm
from lset.kernels.residual_layernorm import residual_layer_norm, fused_residual_layer_norm
from lset.kernels.pool_normalize import fused_pool_normalize
from lset.kernels.rope import apply_rotary_pos_emb, fused_apply_rotary_pos_emb
from lset.kernels.swiglu import swiglu, fused_swiglu
from lset.kernels.geglu import geglu, fused_geglu
from lset.kernels.segment_pool import triton_segment_mean_pool

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
    "triton_segment_mean_pool",
]
