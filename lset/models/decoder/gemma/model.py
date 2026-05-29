"""EmbeddingGemma model — Gemma3 backbone with bidirectional attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.kernels import geglu as _geglu
from lset.kernels import residual_rms_norm as _residual_rms_norm
from lset.kernels import rms_norm as _fused_rms_norm
from lset.kernels.double_residual_rmsnorm import fused_double_residual_rms_norm as _double_rms
from lset.kernels.qk_norm_rope import qk_norm_rope as _fused_qk_norm_rope
from lset.models.decoder.gemma.config import GemmaConfig
from lset.models.decoder.qwen3.attention import apply_rotary_pos_emb


class GemmaRMSNorm(nn.Module):
    """Gemma3 RMSNorm — differs from standard: weight stored as offset, applied as (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fused_rms_norm(x, 1.0 + self.weight, self.eps)


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
    def __init__(self, config: GemmaConfig, is_sliding: bool, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.is_sliding = is_sliding
        self.sliding_window = config.sliding_window
        self.fused_qkv = fused_qkv
        self.attn_scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        if fused_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, q_dim + 2 * kv_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

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

        # Fused QK-norm + RoPE; Gemma uses ``(1 + weight)`` offset semantics.
        q, k = _fused_qk_norm_rope(
            q, k,
            1.0 + self.q_norm.weight,
            1.0 + self.k_norm.weight,
            cos, sin,
            self.q_norm.eps,
        )

        # Custom attention scaling
        q = q * self.attn_scale

        # Native GQA via enable_gqa.
        enable_gqa = q.shape[1] != k.shape[1]

        # Causal attention (matching HF Gemma3 behavior)
        # Note: despite config having use_bidirectional_attention=True,
        # HF's masking utils produce causal masks for all layers.
        if attention_mask is not None and (attention_mask == 0).any():
            causal = (
                torch.triu(
                    torch.full((S, S), float("-inf"), dtype=q.dtype, device=q.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            pad = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
                torch.tensor(float("-inf"), dtype=q.dtype, device=q.device),
            )
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal + pad, scale=1.0, enable_gqa=enable_gqa
            )
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0, enable_gqa=enable_gqa)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)


class GemmaMLP(nn.Module):
    """Gated MLP with GELU-tanh activation (not SiLU)."""

    def __init__(self, config: GemmaConfig, fused_gate_up: bool = False):
        super().__init__()
        self.fused_gate_up = fused_gate_up
        if fused_gate_up:
            self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up:
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            return self.down_proj(_geglu(gate, up))
        return self.down_proj(_geglu(self.gate_proj(x), self.up_proj(x)))


class GemmaBlock(nn.Module):
    """Gemma3 block with 4 layer norms (pre-attn, post-attn, pre-ff, post-ff)."""

    def __init__(self, config: GemmaConfig, is_sliding: bool, fused_projections: bool = False):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config, is_sliding=is_sliding, fused_qkv=fused_projections)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config, fused_gate_up=fused_projections)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention: fuse (residual + hidden_states) + input_layernorm for
        # every block except the first. Gemma RMSNorm uses ``1 + weight``.
        if residual is None:
            attn_in = self.input_layernorm(hidden_states)
            residual = hidden_states
        else:
            attn_in, residual = _residual_rms_norm(
                residual,
                hidden_states,
                1.0 + self.input_layernorm.weight,
                self.input_layernorm.eps,
            )

        attn_out = self.self_attn(attn_in, cos, sin, attention_mask)

        # Fused: post_attention_layernorm(attn_out) → +residual → pre_feedforward_layernorm.
        mlp_in, residual = _double_rms(
            attn_out,
            residual,
            1.0 + self.post_attention_layernorm.weight,
            1.0 + self.pre_feedforward_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        mlp_out = self.mlp(mlp_in)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        # Caller fuses ``residual + mlp_out`` with the next block's
        # input_layernorm (or closes the loop before ``self.norm``).
        return mlp_out, residual


class GemmaEmbeddingModel(nn.Module):
    def __init__(self, config: GemmaConfig, fused_projections: bool = False):
        super().__init__()
        self.config = config
        self.fused_projections = fused_projections
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                GemmaBlock(config, is_sliding=(lt == "sliding_attention"), fused_projections=fused_projections)
                for lt in layer_types
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Two RoPE modules — global (theta=1M) and local (theta=10k)
        self.rotary_emb_global = GemmaRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.rotary_emb_local = GemmaRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_local_base_freq,
        )

        # Track which layers use local vs global RoPE
        self.layer_is_sliding = [lt == "sliding_attention" for lt in layer_types]

        # Gemma3 normalizes embeddings by sqrt(hidden_size)
        self._embed_scale = config.hidden_size**0.5

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

        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            if self.layer_is_sliding[i]:
                cos, sin = cos_local, sin_local
            else:
                cos, sin = cos_global, sin_global
            hidden_states, residual = layer(hidden_states, residual, cos, sin, attention_mask)

        hidden_states = residual + hidden_states
        hidden_states = self.norm(hidden_states)

        return {"hidden_states": hidden_states}
