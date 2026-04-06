from __future__ import annotations

import torch
import torch.nn as nn

from lset.models.decoder.qwen3.attention import Qwen3Attention, Qwen3RMSNorm
from lset.models.decoder.qwen3.config import Qwen3Config
from lset.models.decoder.qwen3.mlp import Qwen3MLP
from lset.kernels import residual_rms_norm as _residual_rms_norm


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, fused_projections: bool = False):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, fused_qkv=fused_projections)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config, fused_gate_up=fused_projections)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Unified forward for both padded and packed modes."""
        packed = cu_seqlens is not None

        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if packed:
            hidden_states = self.self_attn.forward_packed(
                hidden_states, cos, sin, position_ids, cu_seqlens, max_seqlen
            )
        else:
            hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)

        # Fused residual add + post-attention RMSNorm (pre-MLP norm)
        hidden_states, residual = _residual_rms_norm(
            residual, hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Convenience method — routes through forward() for FSDP2 compatibility."""
        return self(
            hidden_states, cos, sin,
            position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
