from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.models.decoder.qwen3.attention import Qwen3RMSNorm
from lset.models.decoder.qwen3.attention import Qwen3RotaryEmbedding
from lset.models.decoder.qwen3.block import Qwen3Block
from lset.models.decoder.qwen3.config import Qwen3Config


class Qwen3Decoder(nn.Module):
    def __init__(self, config: Qwen3Config, fused_projections: bool = False):
        super().__init__()
        self.config = config
        self.fused_projections = fused_projections
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3Block(config, fused_projections=fused_projections) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

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
        """Unified forward supporting both padded and packed modes.

        Padded mode (default): input_ids [B, S], attention_mask [B, S].
        Packed mode: input_ids (T,), position_ids (T,), cu_seqlens (N+1,), max_seqlen int.
        """
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
        """Convenience method — routes through forward() for FSDP2 compatibility."""
        return self(
            input_ids,
            return_lm_logits=return_lm_logits,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def _forward_padded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        return_lm_logits: bool,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        B, S = input_ids.shape

        # Use input_ids.device for RoPE/mask creation since hidden_states
        # may be a DTensor under SequenceParallel
        device = input_ids.device
        cos, sin = self.rotary_emb(S, device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._make_causal_mask(attention_mask, cos.dtype, device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, causal_mask)

        hidden_states = self.norm(hidden_states)

        # When SequenceParallel is active, hidden_states is a Shard(1) DTensor.
        # Pooling needs the full sequence, so gather back to a regular tensor.
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
        hidden_states = self.embed_tokens(input_ids)  # (T, H)

        cos, sin = self.rotary_emb(max_seqlen, hidden_states.device)
        cos = cos.squeeze(0).squeeze(0).to(hidden_states.dtype)  # (max_seqlen, head_dim)
        sin = sin.squeeze(0).squeeze(0).to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)

        result: dict[str, torch.Tensor] = {"hidden_states": hidden_states}
        if return_lm_logits:
            result["lm_logits"] = F.linear(hidden_states, self.embed_tokens.weight)
        return result

    @staticmethod
    def _make_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        B, S = attention_mask.shape
        # Causal mask: upper triangle is -inf [1, 1, S, S]
        causal = (
            torch.triu(torch.full((S, S), float("-inf"), dtype=dtype, device=device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Padding mask: -inf where mask==0, 0 where mask==1 [B, 1, 1, S]
        pad_mask = torch.where(
            attention_mask[:, None, None, :].bool(),
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )
        # Combine: [B, 1, S, S]
        return causal + pad_mask
