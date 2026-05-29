"""Fused ``gate_up_proj + swiglu`` — Triton dual-matmul with a SwiGLU epilogue."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ],
    key=["M", "I", "K"],
)
@triton.jit
def _gate_up_swiglu_fwd_kernel(
    X,
    W,
    OUT,
    GU_SAVE,          # (M, 2I) buffer or dummy
    M, I, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    stride_gu_m, stride_gu_n,
    SAVE_GU: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """One ``(BM, BN)`` output tile; gate and up tiles share the ``x`` load."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(I, BN)
    # L2 swizzle for higher reuse.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    row_mask = rm[:, None] < M
    col_mask = rn[None, :] < I

    # W layout: (2I, K). Gate rows are [0, I); up rows are [I, 2I).
    W_gate = W + rn[None, :] * stride_wn + rk[:, None] * stride_wk
    W_up = W + (rn[None, :] + I) * stride_wn + rk[:, None] * stride_wk
    X_ptr = X + rm[:, None] * stride_xm + rk[None, :] * stride_xk

    acc_gate = tl.zeros((BM, BN), dtype=tl.float32)
    acc_up = tl.zeros((BM, BN), dtype=tl.float32)

    for k_start in range(0, K, BK):
        k_in_range = rk + k_start < K
        a = tl.load(X_ptr, mask=row_mask & k_in_range[None, :], other=0.0)
        wg = tl.load(W_gate, mask=k_in_range[:, None] & col_mask, other=0.0)
        wu = tl.load(W_up, mask=k_in_range[:, None] & col_mask, other=0.0)
        acc_gate += tl.dot(a, wg)
        acc_up += tl.dot(a, wu)
        X_ptr += BK * stride_xk
        W_gate += BK * stride_wk
        W_up += BK * stride_wk

    store_mask = row_mask & col_mask

    # Save pre-SwiGLU activations for backward when training.
    if SAVE_GU:
        gu_base = GU_SAVE + rm[:, None] * stride_gu_m
        tl.store(gu_base + rn[None, :] * stride_gu_n,
                 acc_gate.to(OUT.dtype.element_ty), mask=store_mask)
        tl.store(gu_base + (rn[None, :] + I) * stride_gu_n,
                 acc_up.to(OUT.dtype.element_ty), mask=store_mask)

    # SwiGLU in register file.
    sig = tl.sigmoid(acc_gate)
    out = (acc_gate * sig) * acc_up

    OUT_ptr = OUT + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(OUT_ptr, out.to(OUT.dtype.element_ty), mask=store_mask)


@triton.jit
def _swiglu_grad_kernel(
    DOUT,       # (M, I)
    GU,         # (M, 2I) saved (gate || up)
    DGU,        # (M, 2I) output
    M, I,
    stride_dom, stride_don,
    stride_gu_m, stride_gu_n,
    stride_dgu_m, stride_dgu_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Elementwise SwiGLU backward: write ``[d_gate | d_up]`` into ``dGU`` (M, 2I)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < I)

    dout = tl.load(DOUT + rm[:, None] * stride_dom + rn[None, :] * stride_don,
                   mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GU + rm[:, None] * stride_gu_m + rn[None, :] * stride_gu_n,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GU + rm[:, None] * stride_gu_m + (rn[None, :] + I) * stride_gu_n,
                 mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu_g = gate * sig
    d_up = dout * silu_g
    silu_grad = sig * (1.0 + gate * (1.0 - sig))
    d_gate = dout * up * silu_grad

    out_dtype = DGU.dtype.element_ty
    tl.store(DGU + rm[:, None] * stride_dgu_m + rn[None, :] * stride_dgu_n,
             d_gate.to(out_dtype), mask=mask)
    tl.store(DGU + rm[:, None] * stride_dgu_m + (rn[None, :] + I) * stride_dgu_n,
             d_up.to(out_dtype), mask=mask)


def _fwd(x: Tensor, w: Tensor, save_gu: bool):
    """Fused forward over 2D ``x`` and ``(2I, K)`` ``w``; returns ``(out, gu_or_None)``."""
    assert x.is_cuda and w.is_cuda
    assert x.dim() == 2
    M, K = x.shape
    two_I, K2 = w.shape
    assert K == K2 and two_I % 2 == 0
    I = two_I // 2

    out = torch.empty((M, I), device=x.device, dtype=x.dtype)
    if save_gu:
        gu = torch.empty((M, 2 * I), device=x.device, dtype=x.dtype)
    else:
        gu = torch.empty(1, device=x.device, dtype=x.dtype)  # dummy

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(I, meta["BN"]),)

    _gate_up_swiglu_fwd_kernel[grid](
        x, w, out, gu,
        M, I, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        gu.stride(0) if save_gu else 0, gu.stride(1) if save_gu else 0,
        SAVE_GU=save_gu,
    )
    return out, (gu if save_gu else None)


def _compute_dgu(dout: Tensor, gu: Tensor) -> Tensor:
    M, I = dout.shape
    dgu = torch.empty_like(gu)
    BM, BN = 64, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(I, BN))
    _swiglu_grad_kernel[grid](
        dout, gu, dgu,
        M, I,
        dout.stride(0), dout.stride(1),
        gu.stride(0), gu.stride(1),
        dgu.stride(0), dgu.stride(1),
        BM=BM, BN=BN,
    )
    return dgu


class _FusedGateUpSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, need_backward: bool):
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, orig_shape[-1])
        else:
            x_2d = x
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        out, gu = _fwd(x_2d, weight, save_gu=need_backward)

        if need_backward:
            ctx.save_for_backward(x_2d, weight, gu)
        ctx.orig_shape = orig_shape
        if len(orig_shape) > 2:
            I = weight.shape[0] // 2
            out = out.view(*orig_shape[:-1], I)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight, gu = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        if grad_out.dim() > 2:
            dout = grad_out.reshape(-1, grad_out.shape[-1])
        else:
            dout = grad_out
        if not dout.is_contiguous():
            dout = dout.contiguous()

        # d_gate, d_up via Triton elementwise kernel → (M, 2I) dGU buffer.
        dgu = _compute_dgu(dout, gu)

        # dx = dGU @ W (M, 2I) × (2I, K) → (M, K)
        dx = dgu @ weight
        # dW = dGU.T @ x (2I, M) × (M, K) → (2I, K)
        dW = dgu.t().contiguous() @ x_2d

        if len(orig_shape) > 2:
            dx = dx.view(*orig_shape)
        return dx, dW, None  # None grad for the need_backward bool arg


def fused_gate_up_swiglu(x: Tensor, weight: Tensor) -> Tensor:
    """Returns ``silu(gate) * up`` for ``weight = cat(gate, up)`` (nn.Linear layout)."""
    # Capture grad mode here — inside Function.forward it always reads False.
    need_backward = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    return _FusedGateUpSwiGLUFn.apply(x, weight, need_backward)


# nn.Module wrapper preserving .weight / .in_features / .out_features so TP,
# FP8, and LoRA filters target it like an nn.Linear.


class FusedGateUpSwiGLU(torch.nn.Module):
    """Drop-in for ``nn.Linear(H, 2*I) + chunk + swiglu``; mutually exclusive with LoRA/FP8."""

    def __init__(self, in_features: int, intermediate: int, *, bias: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("FusedGateUpSwiGLU does not support bias")
        self.in_features = in_features
        self.intermediate = intermediate
        self.out_features = 2 * intermediate  # so TP / FP8 / LoRA filters see the right dim
        self.weight = torch.nn.Parameter(torch.empty(2 * intermediate, in_features))
        self._fusion_enabled = True
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def disable_fusion(self) -> None:
        self._fusion_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        # Unwrap DTensor at forward boundary (TP path).
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(w, DTensor):
                w = w.to_local()
        except Exception:
            pass

        if self._fusion_enabled and x.is_cuda:
            return fused_gate_up_swiglu(x, w)

        # Fallback — plain projection + LSET's existing swiglu kernel.
        import torch.nn.functional as F
        from lset.kernels.swiglu import swiglu
        gu = F.linear(x, w)
        gate, up = gu.chunk(2, dim=-1)
        return swiglu(gate, up)
