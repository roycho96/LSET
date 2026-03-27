"""Llama decoder model for embedding.

Llama-Nemotron-Embed is a bidirectional Llama — causal mask removed.
Uses mean pooling (right-padded) instead of Qwen3's last-token pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.models.decoder.qwen3.attention import Qwen3RMSNorm as LlamaRMSNorm
from .attention import LlamaRotaryEmbedding
from .block import LlamaBlock
from .config import LlamaConfig


class LlamaDecoder(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_lm_logits: bool = False,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if cu_seqlens is not None:
            return self._forward_packed(input_ids, position_ids, cu_seqlens, max_seqlen, return_lm_logits)
        return self._forward_padded(input_ids, attention_mask, return_lm_logits)

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        return_lm_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self(
            input_ids, return_lm_logits=return_lm_logits,
            position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )

    def _forward_padded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        return_lm_logits: bool,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        B, S = input_ids.shape
        device = input_ids.device
        cos, sin = self.rotary_emb(S, device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Bidirectional — only create mask if there's actual padding
        attn_mask = None
        if attention_mask is not None and (attention_mask == 0).any():
            attn_mask = self._make_padding_mask(attention_mask, cos.dtype, device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attn_mask)

        hidden_states = self.norm(hidden_states)

        # DTensor → full tensor for pooling
        try:
            from torch.distributed._tensor import DTensor
            if isinstance(hidden_states, DTensor):
                hidden_states = hidden_states.full_tensor()
        except ImportError:
            pass

        result: dict[str, torch.Tensor] = {"hidden_states": hidden_states}
        if return_lm_logits:
            result["lm_logits"] = F.linear(hidden_states, self.embed_tokens.weight)
        return result

    def _forward_packed(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        return_lm_logits: bool,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(max_seqlen, hidden_states.device)
        cos = cos.squeeze(0).squeeze(0).to(hidden_states.dtype)
        sin = sin.squeeze(0).squeeze(0).to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin,
                position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)

        result: dict[str, torch.Tensor] = {"hidden_states": hidden_states}
        if return_lm_logits:
            result["lm_logits"] = F.linear(hidden_states, self.embed_tokens.weight)
        return result

    @staticmethod
    def _make_padding_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Bidirectional padding mask (no causal component)."""
        # [B, 1, 1, S] — -inf for padding, 0 for real tokens
        return torch.where(
            attention_mask[:, None, None, :].bool(),
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )
