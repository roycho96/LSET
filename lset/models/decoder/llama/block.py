"""Llama transformer block — same structure as Qwen3 (pre-norm RMSNorm)."""

from __future__ import annotations

import torch
import torch.nn as nn

from lset.models.decoder.qwen3.attention import Qwen3RMSNorm as LlamaRMSNorm
from .attention import LlamaAttention
from .config import LlamaConfig
from .mlp import LlamaMLP


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

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
        packed = cu_seqlens is not None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if packed:
            hidden_states = self.self_attn.forward_packed(
                hidden_states, cos, sin, position_ids, cu_seqlens, max_seqlen
            )
        else:
            hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        return self(
            hidden_states, cos, sin,
            position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
