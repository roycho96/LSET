"""EmbeddingGemma model — Gemma3 backbone with bidirectional attention.

Key differences from Qwen3/Llama:
- GELU-tanh gated MLP (not SiLU)
- Sliding window attention (512) on most layers, full on every 6th
- QK-norm
- Pre and post feedforward layer norms (4 norms per block)
- query_pre_attn_scalar (custom attention scaling)
- Two different RoPE base frequencies (global vs local)
- Post-pooling linear projection head (768→3072→768)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.models.decoder.qwen3.attention import (
    _rotate_half,
    apply_rotary_pos_emb,
)
from .config import GemmaConfig


class GemmaRMSNorm(nn.Module):
    """Gemma3 RMSNorm — differs from standard: weight stored as offset, applied as (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class GemmaRotaryEmbedding(nn.Module):
    """RoPE with different base frequencies for sliding vs full attention layers."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, is_sliding: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.is_sliding = is_sliding
        self.sliding_window = config.sliding_window
        # Gemma3 uses query_pre_attn_scalar for attention scaling
        self.attn_scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads via repeat_interleave (benchmarked faster than expand+reshape)
        local_q_heads = q.shape[1]
        local_kv_heads = k.shape[1]
        local_kv_groups = local_q_heads // local_kv_heads
        if local_kv_groups > 1:
            k = k.repeat_interleave(local_kv_groups, dim=1)
            v = v.repeat_interleave(local_kv_groups, dim=1)

        # Custom attention scaling
        q = q * self.attn_scale

        # Causal attention (matching HF Gemma3 behavior)
        # Note: despite config having use_bidirectional_attention=True,
        # HF's masking utils produce causal masks for all layers.
        if attention_mask is not None and (attention_mask == 0).any():
            causal = torch.triu(
                torch.full((S, S), float("-inf"), dtype=q.dtype, device=q.device), diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            pad = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
                torch.tensor(float("-inf"), dtype=q.dtype, device=q.device),
            )
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal + pad, scale=1.0)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

class GemmaMLP(nn.Module):
    """Gated MLP with GELU-tanh activation (not SiLU)."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaBlock(nn.Module):
    """Gemma3 block with 4 layer norms (pre-attn, post-attn, pre-ff, post-ff)."""

    def __init__(self, config: GemmaConfig, is_sliding: bool):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config, is_sliding=is_sliding)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Pre-norm MLP with extra norms
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaEmbeddingModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build layers with correct sliding/full attention pattern
        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        self.layers = nn.ModuleList([
            GemmaBlock(config, is_sliding=(lt == "sliding_attention"))
            for lt in layer_types
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Two RoPE modules — global (theta=1M) and local (theta=10k)
        self.rotary_emb_global = GemmaRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.rotary_emb_local = GemmaRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_local_base_freq,
        )

        # Track which layers use local vs global RoPE
        self.layer_is_sliding = [lt == "sliding_attention" for lt in layer_types]

        # Gemma3 normalizes embeddings by sqrt(hidden_size)
        self._embed_scale = config.hidden_size ** 0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids) * self._embed_scale
        B, S = input_ids.shape
        device = input_ids.device

        cos_global, sin_global = self.rotary_emb_global(S, device)
        cos_local, sin_local = self.rotary_emb_local(S, device)
        cos_global = cos_global.to(hidden_states.dtype)
        sin_global = sin_global.to(hidden_states.dtype)
        cos_local = cos_local.to(hidden_states.dtype)
        sin_local = sin_local.to(hidden_states.dtype)

        for i, layer in enumerate(self.layers):
            if self.layer_is_sliding[i]:
                cos, sin = cos_local, sin_local
            else:
                cos, sin = cos_global, sin_global
            hidden_states = layer(hidden_states, cos, sin, attention_mask)

        hidden_states = self.norm(hidden_states)

        return {"hidden_states": hidden_states}
