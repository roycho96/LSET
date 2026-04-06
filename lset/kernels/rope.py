"""Fused Rotary Position Embedding (RoPE) — single Triton kernel.

Standard RoPE (_rotate_half + apply):
  x1, x2 = x[..., :D//2], x[..., D//2:]    # 2 slices
  rotated = cat(-x2, x1)                     # 1 cat + 1 neg
  out = x * cos + rotated * sin              # 2 mul + 1 add
  → 6 kernel launches for Q, same for K = 12 launches per layer

Fused RoPE:
  One kernel reads x, cos, sin; computes both halves in-place; writes out.
  For Q and K simultaneously: 2 kernel launches per layer instead of 12.

Savings: 28 layers × 10 fewer launches = 280 fewer kernel launches per forward pass.
"""

import torch
import triton
import triton.language as tl

# Fixed config — D=128 (head_dim) is constant, so no need for autotune.
# Autotune with variable T causes recompilation for every packed batch.
_BLOCK_HD = 128  # Process full HALF_D (64) in one tile for D=128


# =============================================================================
# Forward Kernel
# =============================================================================

@triton.jit
def _rope_fwd_kernel(
    X,          # [T, H, D] input (Q or K)
    Cos,        # cos values, [T, D] or [T, HALF_D]
    Sin,        # sin values
    Y,          # [T, H, D] output
    T,          # total tokens
    H,          # number of heads
    D: tl.constexpr,          # head dimension (full)
    HALF_D: tl.constexpr,     # D // 2
    stride_xt,  # X token stride
    stride_xh,  # X head stride
    stride_ct,  # Cos token stride
    stride_yt,  # Y token stride
    stride_yh,  # Y head stride
    BLOCK_HD: tl.constexpr,
):
    """Apply RoPE: y1 = x1*c - x2*s, y2 = x2*c + x1*s"""
    pid = tl.program_id(0)
    t_idx = pid // H
    h_idx = pid % H

    if t_idx >= T:
        return

    x_base = t_idx * stride_xt + h_idx * stride_xh
    y_base = t_idx * stride_yt + h_idx * stride_yh
    c_base = t_idx * stride_ct

    for start_d in range(0, HALF_D, BLOCK_HD):
        offs_d = start_d + tl.arange(0, BLOCK_HD)
        mask = offs_d < HALF_D

        x1 = tl.load(X + x_base + offs_d, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X + x_base + HALF_D + offs_d, mask=mask, other=0.0).to(tl.float32)

        c = tl.load(Cos + c_base + offs_d, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(Sin + c_base + offs_d, mask=mask, other=0.0).to(tl.float32)

        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s

        tl.store(Y + y_base + offs_d, y1.to(x1.dtype), mask=mask)
        tl.store(Y + y_base + HALF_D + offs_d, y2.to(x1.dtype), mask=mask)


# =============================================================================
# Backward Kernel
# =============================================================================

@triton.jit
def _rope_bwd_kernel(
    GradY,      # [T, H, D]
    Cos, Sin,
    GradX,      # [T, H, D]
    T, H,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    stride_gyt, stride_gyh,
    stride_ct,
    stride_gxt, stride_gxh,
    BLOCK_HD: tl.constexpr,
):
    """Backward: dx1 = gy1*c + gy2*s, dx2 = -gy1*s + gy2*c"""
    pid = tl.program_id(0)
    t_idx = pid // H
    h_idx = pid % H

    if t_idx >= T:
        return

    gy_base = t_idx * stride_gyt + h_idx * stride_gyh
    gx_base = t_idx * stride_gxt + h_idx * stride_gxh
    c_base = t_idx * stride_ct

    for start_d in range(0, HALF_D, BLOCK_HD):
        offs_d = start_d + tl.arange(0, BLOCK_HD)
        mask = offs_d < HALF_D

        gy1 = tl.load(GradY + gy_base + offs_d, mask=mask, other=0.0).to(tl.float32)
        gy2 = tl.load(GradY + gy_base + HALF_D + offs_d, mask=mask, other=0.0).to(tl.float32)

        c = tl.load(Cos + c_base + offs_d, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(Sin + c_base + offs_d, mask=mask, other=0.0).to(tl.float32)

        gx1 = gy1 * c + gy2 * s
        gx2 = -gy1 * s + gy2 * c

        tl.store(GradX + gx_base + offs_d, gx1.to(gy1.dtype), mask=mask)
        tl.store(GradX + gx_base + HALF_D + offs_d, gx2.to(gy1.dtype), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _get_cos_sin_2d(cos, sin):
    """Normalize cos/sin to [T, D] shape."""
    if cos.dim() == 4:
        # Padded: [1, 1, S, D] → [S, D]
        return cos.squeeze(0).squeeze(0).contiguous(), sin.squeeze(0).squeeze(0).contiguous()
    if cos.dim() == 3:
        # Packed: [T, 1, D] → [T, D]
        return cos.squeeze(1).contiguous(), sin.squeeze(1).contiguous()
    return cos.contiguous(), sin.contiguous()


class FusedRoPE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin):
        is_padded = x.dim() == 4

        if is_padded:
            B, H, S, D = x.shape
            x_3d = x.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
            cos_2d, sin_2d = _get_cos_sin_2d(cos, sin)
            cos_2d = cos_2d[:S].repeat(B, 1).contiguous()
            sin_2d = sin_2d[:S].repeat(B, 1).contiguous()
            T = B * S
        else:
            T, H, D = x.shape
            x_3d = x.contiguous()
            cos_2d, sin_2d = _get_cos_sin_2d(cos, sin)

        HALF_D = D // 2
        y = torch.empty_like(x_3d)

        grid = (T * H,)
        _rope_fwd_kernel[grid](
            x_3d, cos_2d, sin_2d, y,
            T, H, D, HALF_D,
            x_3d.stride(0), x_3d.stride(1),
            cos_2d.stride(0),
            y.stride(0), y.stride(1),
            BLOCK_HD=_BLOCK_HD, num_warps=4,
        )

        ctx.save_for_backward(cos_2d, sin_2d)
        ctx.shape_info = (is_padded, T, H, D)
        if is_padded:
            ctx.padded_shape = (B, H, S, D)
            return y.reshape(B, S, H, D).permute(0, 2, 1, 3)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        cos_2d, sin_2d = ctx.saved_tensors
        is_padded, T, H, D = ctx.shape_info
        HALF_D = D // 2

        if is_padded:
            B, Hh, S, Dd = ctx.padded_shape
            grad_y_3d = grad_y.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        else:
            grad_y_3d = grad_y.contiguous()

        grad_x = torch.empty_like(grad_y_3d)

        grid = (T * H,)
        _rope_bwd_kernel[grid](
            grad_y_3d, cos_2d, sin_2d, grad_x,
            T, H, D, HALF_D,
            grad_y_3d.stride(0), grad_y_3d.stride(1),
            cos_2d.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            BLOCK_HD=_BLOCK_HD, num_warps=4,
        )

        if is_padded:
            grad_x = grad_x.reshape(B, S, H, D).permute(0, 2, 1, 3)
        return grad_x, None, None


# =============================================================================
# Public API
# =============================================================================

_FUSED_ROPE_THRESHOLD = 128


def fused_apply_rotary_pos_emb(q, k, cos, sin):
    """Apply fused RoPE to both Q and K."""
    q_out = FusedRoPE.apply(q, cos, sin)
    k_out = FusedRoPE.apply(k, cos, sin)
    return q_out, k_out


def apply_rotary_pos_emb(q, k, cos, sin):
    """RoPE with automatic Triton dispatch."""
    T = q.shape[0] if q.dim() == 3 else q.shape[0] * q.shape[2]
    if q.is_cuda and T >= _FUSED_ROPE_THRESHOLD:
        return fused_apply_rotary_pos_emb(q, k, cos, sin)
    # Fallback
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
