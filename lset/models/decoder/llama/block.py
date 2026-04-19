"""Llama transformer block — same structure as Qwen3 (pre-norm RMSNorm).

Returns ``(mlp_out, residual)`` so ``LlamaDecoder`` can fuse the block-boundary
residual add with the next block's ``input_layernorm``. See ``Qwen3Block`` for
the full rationale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lset.kernels import residual_rms_norm as _residual_rms_norm
from lset.models.decoder.llama.attention import LlamaAttention
from lset.models.decoder.llama.config import LlamaConfig
from lset.models.decoder.llama.mlp import LlamaMLP
from lset.models.decoder.qwen3.attention import Qwen3RMSNorm as LlamaRMSNorm


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, fused_projections: bool = False):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, fused_qkv=fused_projections)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config, fused_gate_up=fused_projections)

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

        mlp_in, residual = _residual_rms_norm(
            residual,
            attn_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        mlp_out = self.mlp(mlp_in)
        return mlp_out, residual
