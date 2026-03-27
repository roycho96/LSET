"""BERT / XLM-RoBERTa encoder model.

Supports both BERT and XLM-RoBERTa architectures:
- Bidirectional attention (no causal mask)
- Absolute position embeddings (with optional offset for RoBERTa)
- Post-norm LayerNorm (after residual, not before)
- GELU MLP (fc1 → GELU → fc2, not SwiGLU)
- Bias in all projections
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BertConfig


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.position_offset = config.position_offset

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0) + self.position_offset
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(embeddings)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.query(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Bidirectional — no causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        # Post-norm: residual + dense + LayerNorm
        return self.LayerNorm(hidden_states + self.dense(attn_out))


class BertMLP(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.gelu(self.dense_in(hidden_states))
        hidden_states = self.dense_out(hidden_states)
        return self.LayerNorm(residual + hidden_states)


class BertBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if cu_seqlens is not None:
            return self._forward_packed(input_ids, position_ids, cu_seqlens, max_seqlen)
        return self._forward_padded(input_ids, attention_mask, token_type_ids)

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> dict[str, torch.Tensor]:
        return self(
            input_ids,
            position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )

    def _forward_padded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids)

        attn_mask = None
        if attention_mask is not None and (attention_mask == 0).any():
            attn_mask = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device),
                torch.tensor(float("-inf"), dtype=hidden_states.dtype, device=hidden_states.device),
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)

        return {"hidden_states": hidden_states}

    def _forward_packed(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> dict[str, torch.Tensor]:
        """Packed forward for BERT encoder using flash_attn with causal=False."""
        T = input_ids.shape[0]

        # Compute embeddings per-token (no batch dimension)
        word_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(position_ids + self.embeddings.position_offset)
        type_emb = self.embeddings.token_type_embeddings(
            torch.zeros(T, dtype=torch.long, device=input_ids.device)
        )
        hidden_states = self.embeddings.LayerNorm(word_emb + pos_emb + type_emb)

        # Build block-diagonal mask for SDPA (bidirectional)
        mask = torch.full((T, T), float("-inf"), dtype=hidden_states.dtype, device=hidden_states.device)
        for i in range(cu_seqlens.shape[0] - 1):
            s = int(cu_seqlens[i])
            e = int(cu_seqlens[i + 1])
            mask[s:e, s:e] = 0.0
        attn_mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = hidden_states.unsqueeze(0)  # (1, T, H)
            hidden_states = layer(hidden_states, attn_mask)
            hidden_states = hidden_states.squeeze(0)  # (T, H)

        return {"hidden_states": hidden_states}
