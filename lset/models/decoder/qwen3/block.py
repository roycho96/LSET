from __future__ import annotations

import torch
import torch.nn as nn

from lset.kernels import residual_rms_norm as _residual_rms_norm
from lset.models.decoder.qwen3.attention import Qwen3Attention
from lset.models.decoder.qwen3.attention import Qwen3RMSNorm
from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.mlp import Qwen3MLP


class Qwen3Block(nn.Module):
    """Qwen3 transformer block returning ``(mlp_out, residual)``."""

    def __init__(self, config: Qwen3Config, fused_projections: bool = False):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, fused_qkv=fused_projections)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config, fused_gate_up=fused_projections)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = cu_seqlens is not None

        # Pre-attention: fuse (residual + hidden_states) + input_layernorm for
        # every block except the first.
        if residual is None:
            attn_in = self.input_layernorm(hidden_states)
            residual = hidden_states
        else:
            attn_in, residual = _residual_rms_norm(
                residual,
                hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )

        if packed:
            attn_out = self.self_attn.forward_packed(
                attn_in, cos, sin, position_ids, cu_seqlens, max_seqlen
            )
        else:
            attn_out = self.self_attn(attn_in, cos, sin, attention_mask)

        # Post-attention: fused residual add + post-attention RMSNorm.
        mlp_in, residual = _residual_rms_norm(
            residual,
            attn_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        mlp_out = self.mlp(mlp_in)
        # No final residual add here — the next block (or the decoder's
        # post-loop add) closes the loop.
        return mlp_out, residual
