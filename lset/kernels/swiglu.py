"""Fused SwiGLU activation — single Triton kernel for silu(gate) * up."""

import torch
import triton
import triton.language as tl

from torch import Tensor

# Fixed block size — elementwise kernels don't benefit from autotuning on N.
# Using autotune with key=["N"] causes recompilation for every different input
# size, which is catastrophic with variable-length packed sequences.
_BLOCK_SIZE = 2048
_NUM_WARPS = 8


# =============================================================================
# Forward Kernel
# =============================================================================


@triton.jit
def _swiglu_fwd_kernel(
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

    # SiLU(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(Out + offs, out.to(gate.dtype), mask=mask)


# =============================================================================
# Backward Kernel
# =============================================================================


@triton.jit
def _swiglu_bwd_kernel(
    GradOut,  # [total] upstream gradient
    Gate,  # [total] gate_proj output (saved from forward)
    Up,  # [total] up_proj output (saved from forward)
    GradGate,  # [total] gradient for gate
    GradUp,  # [total] gradient for up
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for SwiGLU."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    grad_out = tl.load(GradOut + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)

    sig_gate = tl.sigmoid(gate)
    silu_gate = gate * sig_gate

    grad_up = grad_out * silu_gate
    grad_gate = grad_out * up * sig_gate * (1.0 + gate * (1.0 - sig_gate))

    tl.store(GradGate + offs, grad_gate.to(gate.dtype), mask=mask)
    tl.store(GradUp + offs, grad_up.to(gate.dtype), mask=mask)


# =============================================================================
# Autograd Function
# =============================================================================


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        N = gate_c.numel()

        grid = (triton.cdiv(N, _BLOCK_SIZE),)
        _swiglu_fwd_kernel[grid](gate_c, up_c, out, N, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS)

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
        _swiglu_bwd_kernel[grid](
            grad_out_c, gate, up, grad_gate, grad_up, N, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS
        )

        return grad_gate, grad_up


# =============================================================================
# Public API
# =============================================================================

_FUSED_SWIGLU_THRESHOLD = 4096


def fused_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Fused SwiGLU: silu(gate) * up in a single kernel."""
    return FusedSwiGLU.apply(gate, up)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """SwiGLU with automatic Triton dispatch."""
    if gate.is_cuda and gate.numel() >= _FUSED_SWIGLU_THRESHOLD:
        return fused_swiglu(gate, up)
    return torch.nn.functional.silu(gate) * up
