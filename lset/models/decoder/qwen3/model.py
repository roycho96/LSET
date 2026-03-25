from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Qwen3RMSNorm, Qwen3RotaryEmbedding
from .block import Qwen3Block
from .config import Qwen3Config


class Qwen3Decoder(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_lm_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        B, S = input_ids.shape

        cos, sin = self.rotary_emb(S, hidden_states.device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Build causal + padding mask for SDPA
        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._make_causal_mask(attention_mask, hidden_states.dtype, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, causal_mask)

        hidden_states = self.norm(hidden_states)

        result: dict[str, torch.Tensor] = {"hidden_states": hidden_states}
        if return_lm_logits:
            # Tied embeddings: reuse embed_tokens weight as lm_head
            result["lm_logits"] = F.linear(hidden_states, self.embed_tokens.weight)
        return result

    @staticmethod
    def _make_causal_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        B, S = attention_mask.shape
        # Causal mask: upper triangle is -inf [1, 1, S, S]
        causal = torch.triu(
            torch.full((S, S), float("-inf"), dtype=dtype, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        # Padding mask: -inf where mask==0, 0 where mask==1 [B, 1, 1, S]
        pad_mask = torch.where(
            attention_mask[:, None, None, :].bool(),
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )
        # Combine: [B, 1, S, S]
        return causal + pad_mask
