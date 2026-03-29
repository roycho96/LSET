from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from lset.kernels import rms_norm as _fused_rms_norm
from lset.kernels import apply_rotary_pos_emb as _fused_apply_rotary_pos_emb


# Global attention backend setting: "auto" | "flash_attn" | "varlen_attn" | "sdpa"
_ATTN_BACKEND = "auto"


def set_attn_backend(backend: str):
    """Set the global attention backend for packed mode."""
    global _ATTN_BACKEND
    assert backend in ("auto", "flash_attn", "varlen_attn", "sdpa"), \
        f"Invalid backend: {backend}"
    _ATTN_BACKEND = backend


def get_attn_backend() -> str:
    return _ATTN_BACKEND


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [S, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, S, head_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, S, head_dim]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_apply_rotary_pos_emb(q, k, cos, sin)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.fused_qkv = fused_qkv

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        if fused_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, q_dim + 2 * kv_dim, bias=False)
            # Split sizes for after projection (adjusted for TP via -1 based calculation)
            self._q_dim = q_dim
            self._kv_dim = kv_dim
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
            # Split using local sizes (TP shards the output dim proportionally)
            total_dim = qkv.shape[-1]
            # Ratio: q_dim : kv_dim : kv_dim = num_heads : num_kv_heads : num_kv_heads
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

        # Use SDPA with causal mask or custom mask
        if attention_mask is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        """Packed forward path.

        Args:
            hidden_states: (total_tokens, H)
            cos, sin: (max_seqlen, head_dim)
            position_ids: (total_tokens,)
            cu_seqlens: (num_seqs + 1,) int32
            max_seqlen: int
        """
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

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Position-indexed RoPE
        cos_pos = cos[position_ids].unsqueeze(1)  # (T, 1, D)
        sin_pos = sin[position_ids].unsqueeze(1)
        q, k = _fused_apply_rotary_pos_emb(q, k, cos_pos, sin_pos)

        local_q_heads = q.shape[1]
        local_kv_heads = k.shape[1]
        local_kv_groups = local_q_heads // local_kv_heads
        attn_out = _flash_or_sdpa_packed(
            q, k, v, cu_seqlens, max_seqlen,
            local_q_heads, local_kv_heads, local_kv_groups,
        )

        attn_out = attn_out.reshape(T, -1)
        return self.o_proj(attn_out)


def _try_varlen_attn(q, k, v, cu_seqlens, max_seqlen, causal=True):
    """Try PyTorch native varlen_attn. Requires CUDA + fp16/bf16."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    try:
        from torch.nn.attention.varlen import varlen_attn
        window = (-1, 0) if causal else (-1, -1)
        return varlen_attn(q, k, v, cu_seqlens, cu_seqlens,
                          max_seqlen, max_seqlen, window_size=window)
    except ImportError:
        return None


def _try_flash_attn(q, k, v, cu_seqlens, max_seqlen, causal=True):
    """Try flash_attn_varlen_func. Requires CUDA + fp16/bf16."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    try:
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
        )
    except ImportError:
        return None


def _flash_or_sdpa_packed(q, k, v, cu_seqlens, max_seqlen,
                          num_heads, num_kv_heads, num_kv_groups,
                          causal=True):
    """Configurable attention backend for packed sequences.

    Backend selection ("auto" strategy):
      flash_attn (v2.8.3+) compiles without graph breaks, so it's preferred.
      varlen_attn is PyTorch-native and also compile-friendly.
      SDPA fallback builds O(T^2) block-diagonal mask.

    Priority: flash_attn > varlen_attn > SDPA (when auto).
    """
    backend = _ATTN_BACKEND

    if backend == "auto":
        # flash_attn is fastest and compiles cleanly (FA 2.8.3+)
        result = _try_flash_attn(q, k, v, cu_seqlens, max_seqlen, causal)
        if result is not None:
            return result
        result = _try_varlen_attn(q, k, v, cu_seqlens, max_seqlen, causal)
        if result is not None:
            return result
    elif backend == "flash_attn":
        result = _try_flash_attn(q, k, v, cu_seqlens, max_seqlen, causal)
        if result is not None:
            return result
    elif backend == "varlen_attn":
        result = _try_varlen_attn(q, k, v, cu_seqlens, max_seqlen, causal)
        if result is not None:
            return result

    return _sdpa_packed_fallback(
        q, k, v, cu_seqlens, max_seqlen,
        num_heads, num_kv_heads, num_kv_groups,
        causal=causal,
    )


def _sdpa_packed_fallback(q, k, v, cu_seqlens, max_seqlen,
                          num_heads, num_kv_heads, num_kv_groups,
                          causal=True):
    """SDPA fallback for packed sequences when flash_attn is unavailable.

    Builds a block-diagonal mask and runs SDPA on (1, T, ...) tensors.
    When causal=True, applies causal (lower triangular) masking within each block.
    When causal=False (encoder-style), applies full bidirectional attention within blocks.
    """
    T = q.shape[0]
    device = q.device
    dtype = q.dtype

    # Build block-diagonal mask: (T, T)
    # Each sequence attends only to itself
    mask = torch.full((T, T), float("-inf"), dtype=dtype, device=device)
    for i in range(cu_seqlens.shape[0] - 1):
        s = int(cu_seqlens[i])
        e = int(cu_seqlens[i + 1])
        seq_len = e - s
        if causal:
            # Causal within this sequence block
            causal_block = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
                diagonal=1,
            )
            mask[s:e, s:e] = causal_block
        else:
            # Full bidirectional attention within block
            mask[s:e, s:e] = 0.0

    attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    # Reshape to (1, T, num_heads, head_dim) → (1, num_heads, T, head_dim)
    q = q.unsqueeze(0).transpose(1, 2)  # (1, num_heads, T, D)
    k = k.unsqueeze(0).transpose(1, 2)  # (1, num_kv_heads, T, D)
    v = v.unsqueeze(0).transpose(1, 2)

    # GQA: repeat KV heads via repeat_interleave (benchmarked faster than expand+reshape)
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # (1, num_heads, T, D) → (T, num_heads, D)
    attn_out = attn_out.squeeze(0).transpose(0, 1)
    return attn_out


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fused_rms_norm(x, self.weight, self.eps)
