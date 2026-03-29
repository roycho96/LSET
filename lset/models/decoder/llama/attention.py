"""Llama attention — same as Qwen3 but without QK-norm, with Llama3 RoPE scaling.

Reuses Qwen3 RMSNorm and packed attention infrastructure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.models.decoder.qwen3.attention import (
    Qwen3RMSNorm as LlamaRMSNorm,
    apply_rotary_pos_emb,
    _rotate_half,
    _flash_or_sdpa_packed,
)
from .config import LlamaConfig


class LlamaRotaryEmbedding(nn.Module):
    """RoPE with optional Llama3 frequency scaling."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float,
                 rope_scaling: dict | None = None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            inv_freq = self._apply_llama3_scaling(
                inv_freq, rope_scaling,
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    @staticmethod
    def _apply_llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
        factor = scaling["factor"]
        low_freq_factor = scaling.get("low_freq_factor", 1.0)
        high_freq_factor = scaling.get("high_freq_factor", 4.0)
        old_context_len = scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        new_inv_freq = inv_freq.clone()
        for i in range(len(inv_freq)):
            freq = inv_freq[i].item()
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                pass  # High freq: keep as is
            elif wavelen > low_freq_wavelen:
                new_inv_freq[i] = freq / factor  # Low freq: scale down
            else:
                # Smooth interpolation
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_inv_freq[i] = (1 - smooth) * freq / factor + smooth * freq
        return new_inv_freq

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.fused_qkv = fused_qkv

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        bias = config.attention_bias
        if fused_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, q_dim + 2 * kv_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=bias)
            self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=bias)
            self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=bias)
        # No QK-norm (key difference from Qwen3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        if self.fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            total_dim = qkv.shape[-1]
            ratio = self.num_heads + 2 * self.num_kv_heads
            local_q_dim = total_dim * self.num_heads // ratio
            local_kv_dim = total_dim * self.num_kv_heads // ratio
            q, k, v = qkv.split([local_q_dim, local_kv_dim, local_kv_dim], dim=-1)
            q = q.view(B, S, -1, self.head_dim).transpose(1, 2)
            k = k.view(B, S, -1, self.head_dim).transpose(1, 2)
            v = v.view(B, S, -1, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads via repeat_interleave (benchmarked faster than expand+reshape)
        local_q_heads = q.shape[1]
        local_kv_heads = k.shape[1]
        local_kv_groups = local_q_heads // local_kv_heads
        if local_kv_groups > 1:
            k = k.repeat_interleave(local_kv_groups, dim=1)
            v = v.repeat_interleave(local_kv_groups, dim=1)

        # Bidirectional — no causal mask, just padding mask
        if attention_mask is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        T, _ = hidden_states.shape
        if self.fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            total_dim = qkv.shape[-1]
            ratio = self.num_heads + 2 * self.num_kv_heads
            local_q_dim = total_dim * self.num_heads // ratio
            local_kv_dim = total_dim * self.num_kv_heads // ratio
            q, k, v = qkv.split([local_q_dim, local_kv_dim, local_kv_dim], dim=-1)
            q = q.view(T, -1, self.head_dim)
            k = k.view(T, -1, self.head_dim)
            v = v.view(T, -1, self.head_dim)
        else:
            q = self.q_proj(hidden_states).view(T, -1, self.head_dim)
            k = self.k_proj(hidden_states).view(T, -1, self.head_dim)
            v = self.v_proj(hidden_states).view(T, -1, self.head_dim)

        cos_pos = cos[position_ids].unsqueeze(1)
        sin_pos = sin[position_ids].unsqueeze(1)
        q = (q * cos_pos) + (_rotate_half(q) * sin_pos)
        k = (k * cos_pos) + (_rotate_half(k) * sin_pos)

        local_q_heads = q.shape[1]
        local_kv_heads = k.shape[1]
        local_kv_groups = local_q_heads // local_kv_heads
        attn_out = _flash_or_sdpa_packed(
            q, k, v, cu_seqlens, max_seqlen,
            local_q_heads, local_kv_heads, local_kv_groups,
            causal=False,
        )

        attn_out = attn_out.reshape(T, -1)
        return self.o_proj(attn_out)
