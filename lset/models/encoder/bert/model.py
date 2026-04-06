"""BERT / XLM-RoBERTa encoder model.

Supports both BERT and XLM-RoBERTa architectures:
- Bidirectional attention (no causal mask)
- Absolute position embeddings (with optional offset for RoBERTa)
- Post-norm LayerNorm (after residual, not before)
- GELU MLP (fc1 → GELU → fc2, not SwiGLU)
- Bias in all projections

Fused kernel optimizations:
- FusedLayerNorm (embedding layer norm)
- FusedResidualLayerNorm (post-norm in attention and MLP)
- Fused QKV projection (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.kernels import layer_norm as _fused_layer_norm
from lset.kernels import residual_layer_norm as _residual_layer_norm
from lset.models.encoder.bert.config import BertConfig


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
        return _fused_layer_norm(
            embeddings,
            self.LayerNorm.weight,
            self.LayerNorm.bias,
            self.LayerNorm.eps,
        )


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.fused_qkv = fused_qkv
        self.layer_norm_eps = config.layer_norm_eps

        if fused_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        else:
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

        if self.fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.query(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Bidirectional — no causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        # Post-norm: fused residual + LayerNorm
        normed, _ = _residual_layer_norm(
            hidden_states,
            self.dense(attn_out),
            self.LayerNorm.weight,
            self.LayerNorm.bias,
            self.layer_norm_eps,
        )
        return normed


class BertMLP(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_eps = config.layer_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.gelu(self.dense_in(hidden_states))
        hidden_states = self.dense_out(hidden_states)
        # Post-norm: fused residual + LayerNorm
        normed, _ = _residual_layer_norm(
            residual,
            hidden_states,
            self.LayerNorm.weight,
            self.LayerNorm.bias,
            self.layer_norm_eps,
        )
        return normed


class BertBlock(nn.Module):
    def __init__(self, config: BertConfig, fused_qkv: bool = False):
        super().__init__()
        self.attention = BertAttention(config, fused_qkv=fused_qkv)
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
    def __init__(self, config: BertConfig, fused_projections: bool = False):
        super().__init__()
        self.config = config
        self.fused_projections = fused_projections
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList(
            [BertBlock(config, fused_qkv=fused_projections) for _ in range(config.num_hidden_layers)]
        )

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
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
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
        """Packed forward for BERT encoder.

        Uses varlen_attn or flash_attn with causal=False when available,
        falling back to O(T^2) SDPA block-diagonal mask.
        """
        T = input_ids.shape[0]

        # Compute embeddings per-token (no batch dimension)
        word_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(position_ids + self.embeddings.position_offset)
        type_emb = self.embeddings.token_type_embeddings(torch.zeros(T, dtype=torch.long, device=input_ids.device))
        hidden_states = _fused_layer_norm(
            word_emb + pos_emb + type_emb,
            self.embeddings.LayerNorm.weight,
            self.embeddings.LayerNorm.bias,
            self.embeddings.LayerNorm.eps,
        )

        # Try to use efficient varlen backend for packed bidirectional attention
        use_varlen = self._can_use_varlen(hidden_states)

        if use_varlen:
            for layer in self.layers:
                hidden_states = self._forward_packed_layer_varlen(
                    layer,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                )
        else:
            # Fallback: O(T^2) block-diagonal mask
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

    @staticmethod
    def _can_use_varlen(hidden_states: torch.Tensor) -> bool:
        """Check if efficient varlen backends are available."""
        if not hidden_states.is_cuda:
            return False
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            from torch.nn.attention.varlen import varlen_attn  # noqa: F401

            return True
        except ImportError:
            pass
        return False

    def _forward_packed_layer_varlen(
        self,
        layer,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Forward a single BERT layer using varlen attention (bidirectional)."""
        attn = layer.attention
        T = hidden_states.shape[0]
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        if attn.fused_qkv:
            qkv = attn.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(T, num_heads, head_dim)
            k = k.view(T, num_heads, head_dim)
            v = v.view(T, num_heads, head_dim)
        else:
            q = attn.query(hidden_states).view(T, num_heads, head_dim)
            k = attn.key(hidden_states).view(T, num_heads, head_dim)
            v = attn.value(hidden_states).view(T, num_heads, head_dim)

        # Use flash_attn or varlen_attn with causal=False
        attn_out = None
        try:
            from flash_attn import flash_attn_varlen_func

            attn_out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=False,
            )
        except ImportError:
            pass

        if attn_out is None:
            from torch.nn.attention.varlen import varlen_attn

            attn_out = varlen_attn(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                window_size=(-1, -1),  # bidirectional
            )

        attn_out = attn_out.reshape(T, -1)
        # Post-norm: fused residual + LayerNorm
        hidden_states, _ = _residual_layer_norm(
            hidden_states,
            attn.dense(attn_out),
            attn.LayerNorm.weight,
            attn.LayerNorm.bias,
            attn.layer_norm_eps,
        )

        # MLP with post-norm (uses fused residual layer norm internally)
        hidden_states = layer.mlp(hidden_states)

        return hidden_states
