"""Fused GeGLU activation — single Triton kernel for gelu_tanh(gate) * up."""

import math

import torch
import triton
import triton.language as tl

from torch import Tensor

_BLOCK_SIZE = 2048
_NUM_WARPS = 8
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_TANH_COEFF = 0.044715


# =============================================================================
# Forward Kernel
# =============================================================================


@triton.jit
def _geglu_fwd_kernel(
    Gate,  # [total] flattened gate_proj output
    Up,  # [total] flattened up_proj output
    Out,  # [total] output
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = gate * gate * gate
    inner = 0.7978845608028654 * (gate + 0.044715 * x3)  # sqrt(2/pi) ≈ 0.7978845608
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    gelu_gate = 0.5 * gate * (1.0 + tanh_inner)
    out = gelu_gate * up

    tl.store(Out + offs, out.to(gate.dtype), mask=mask)


# =============================================================================
# Backward Kernel
# =============================================================================


@triton.jit
def _geglu_bwd_kernel(
    GradOut,  # [total] upstream gradient
    Gate,  # [total] gate_proj output (saved from forward)
    Up,  # [total] up_proj output (saved from forward)
    GradGate,  # [total] gradient for gate
    GradUp,  # [total] gradient for up
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for GeGLU."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    grad_out = tl.load(GradOut + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)

    # Forward recomputation
    x3 = gate * gate * gate
    inner = 0.7978845608028654 * (gate + 0.044715 * x3)
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    gelu_gate = 0.5 * gate * (1.0 + tanh_inner)

    # d_gelu_tanh/d_gate
    sech2 = 1.0 - tanh_inner * tanh_inner  # sech^2(inner)
    d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * gate * gate)
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * gate * sech2 * d_inner

    grad_up = grad_out * gelu_gate
    grad_gate = grad_out * up * d_gelu

    tl.store(GradGate + offs, grad_gate.to(gate.dtype), mask=mask)
    tl.store(GradUp + offs, grad_up.to(gate.dtype), mask=mask)


# =============================================================================
# Autograd Function
# =============================================================================


class FusedGeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        N = gate_c.numel()

        grid = (triton.cdiv(N, _BLOCK_SIZE),)
        _geglu_fwd_kernel[grid](gate_c, up_c, out, N, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS)

        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        gate, up = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        N = gate.numel()

        grid = (triton.cdiv(N, _BLOCK_SIZE),)
        _geglu_bwd_kernel[grid](
            grad_out_c, gate, up, grad_gate, grad_up, N, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS
        )

        return grad_gate, grad_up


# =============================================================================
# Public API
# =============================================================================

_FUSED_GEGLU_THRESHOLD = 4096


def fused_geglu(gate: Tensor, up: Tensor) -> Tensor:
    """Fused GeGLU: gelu_tanh(gate) * up in a single kernel."""
    return FusedGeGLU.apply(gate, up)


def geglu(gate: Tensor, up: Tensor) -> Tensor:
    """GeGLU with automatic Triton dispatch."""
    if gate.is_cuda and gate.numel() >= _FUSED_GEGLU_THRESHOLD:
        return fused_geglu(gate, up)
    return torch.nn.functional.gelu(gate, approximate="tanh") * up
