"""
SM100 (Blackwell) Fused Contrastive Loss Kernel

CUDA C++ kernel using SM100-specific optimizations for the fused contrastive
loss Q@K^T tiled computation. Uses multi-stage async pipeline, optimized tile
sizes, and SM100's larger shared memory / register file.

Architecture:
  - Forward: tiled Q@K^T with online LogSumExp (same algorithm as Triton kernel)
  - Backward: dQ and dK kernels with score recomputation
  - SM100-only: graceful fallback to Triton kernel on older GPUs

Public API matches loss.py's FusedDenseLoss interface.
"""

import os
import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

logger = logging.getLogger(__name__)

# Loss type constants (must match CUDA side)
LOSS_MULTI = 0
LOSS_SOFT = 1
LOSS_CROSS = 2

_LOSS_TYPE_MAP = {"multi": LOSS_MULTI, "soft": LOSS_SOFT, "cross": LOSS_CROSS}

# LSE modes
LSE_NEG_ONLY = 0
LSE_VALID_ALL = 1

# Extension singleton
_sm100_ext = None
_sm100_available = None


def _load_sm100_extension():
    """JIT compile and load the SM100 CUDA extension."""
    global _sm100_ext
    if _sm100_ext is not None:
        return _sm100_ext

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

        # Detect GPU architecture for compile flags
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch_flag = f"-arch=sm_{major * 10 + minor}"
        else:
            arch_flag = "-arch=sm_120"

        # Find Python include dirs for compilation
        import sysconfig
        extra_includes = []
        py_include = sysconfig.get_path("include")
        plat_include = sysconfig.get_path("platinclude")
        for candidate in [py_include, plat_include]:
            if candidate and os.path.isdir(candidate):
                extra_includes.append(candidate)
            # WSL mounts may remap /usr → /mnt/wsl/ubuntu/usr
            if candidate and candidate.startswith("/usr/"):
                wsl = candidate.replace("/usr/", "/mnt/wsl/ubuntu/usr/", 1)
                if os.path.isdir(wsl) and wsl not in extra_includes:
                    extra_includes.append(wsl)
        # pyconfig.h on Debian includes <MULTIARCH/pythonX.Y/pyconfig.h>
        # relative to /usr/include — add that WSL-remapped path
        multiarch = sysconfig.get_config_var("MULTIARCH")
        pyver = sysconfig.get_config_var("py_version_short")
        if multiarch and pyver:
            ma_inc = f"/mnt/wsl/ubuntu/usr/include/{multiarch}/python{pyver}"
            if os.path.isdir(ma_inc) and ma_inc not in extra_includes:
                extra_includes.append(ma_inc)

        # On WSL, pyconfig.h includes <x86_64-linux-gnu/.../pyconfig.h>
        # relative to /usr/include. Add the WSL-remapped path for the host
        # compiler (bindings.cpp) but NOT for nvcc (conflicts with CCCL).
        host_sys_flags = []
        wsl_usr_inc = "/mnt/wsl/ubuntu/usr/include"
        if os.path.isdir(wsl_usr_inc):
            host_sys_flags = [f"-isystem{wsl_usr_inc}"]

        # CUTLASS include path (for CuTe headers)
        cutlass_inc = os.path.expanduser("~/workspace/cutlass/include")
        if os.path.isdir(cutlass_inc):
            extra_includes.append(cutlass_inc)

        _sm100_ext = load(
            name="loss_sm100",
            sources=[
                os.path.join(csrc_dir, "loss_sm100.cu"),
                os.path.join(csrc_dir, "bindings.cpp"),
            ],
            extra_cflags=host_sys_flags,
            extra_cuda_cflags=[
                arch_flag,
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
            extra_include_paths=extra_includes,
            verbose=False,
        )
        logger.info("SM100 fused loss extension compiled successfully")
        return _sm100_ext
    except Exception as e:
        logger.warning(f"Failed to compile SM100 fused loss extension: {e}")
        return None


def is_sm100_available() -> bool:
    """Check if SM100+ GPU is available and extension can be loaded."""
    global _sm100_available
    if _sm100_available is not None:
        return _sm100_available

    if not torch.cuda.is_available():
        _sm100_available = False
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        _sm100_available = False
        return False

    ext = _load_sm100_extension()
    if ext is None:
        _sm100_available = False
        return False

    try:
        _sm100_available = ext.is_sm100()
    except Exception:
        _sm100_available = False

    return _sm100_available


def _neg_lse_forward_sm100(q_scaled, k, labels):
    """SM100: compute logsumexp of negative-only scores."""
    ext = _load_sm100_extension()
    out_lse, _ = ext.fwd(q_scaled.contiguous(), k.contiguous(),
                         labels.contiguous(), 1.0, LSE_NEG_ONLY)
    return out_lse


def _all_lse_forward_sm100(q_scaled, k, labels):
    """SM100: compute logsumexp of all valid scores."""
    ext = _load_sm100_extension()
    out_lse, _ = ext.fwd(q_scaled.contiguous(), k.contiguous(),
                         labels.contiguous(), 1.0, LSE_VALID_ALL)
    return out_lse


def _backward_sm100(q_scaled, k, labels, ref_lse, aux, w, loss_type_int):
    """SM100: compute dQ and dK gradients."""
    ext = _load_sm100_extension()
    dq, dk = ext.bwd(
        q_scaled.contiguous(), k.contiguous(), labels.contiguous(),
        ref_lse.contiguous(), aux.contiguous(), w.contiguous(),
        loss_type_int,
    )
    return dq, dk


def _resolve_positive_pairs(q_scaled, k, labels, pos_qi, pos_di, pos_counts, neg_counts):
    """Prepare positive pair indices (same logic as loss.py)."""
    num_queries = q_scaled.shape[0]
    num_docs = k.shape[0]
    device = q_scaled.device

    if pos_qi is not None and pos_di is not None:
        pos_qi = pos_qi.to(device, non_blocking=True)
        pos_di = pos_di.to(device, non_blocking=True)
        num_pos = (
            pos_counts.to(device, non_blocking=True).to(torch.int64)
            if pos_counts is not None
            else torch.bincount(pos_qi, minlength=num_queries).to(torch.int64)
        )
    else:
        pos_mask = labels > 0
        num_pos = pos_mask.sum(dim=1)
        pos_qi, pos_di = torch.where(pos_mask)

    if neg_counts is not None:
        has_neg = neg_counts.to(device, non_blocking=True).to(torch.int64) > 0
    else:
        has_neg = num_pos < num_docs

    with torch.no_grad():
        pos_scores = (q_scaled[pos_qi] * k[pos_di]).sum(dim=1, dtype=torch.float32)
        pos_label_values = labels[pos_qi, pos_di].to(torch.float32)

    return pos_qi, pos_di, num_pos, has_neg, pos_scores, pos_label_values


class FusedDenseLossSM100(torch.autograd.Function):
    """SM100-optimized fused contrastive loss via CUDA C++ extension."""

    @staticmethod
    def forward(
        ctx,
        q_scaled: Tensor,
        k: Tensor,
        labels: Tensor,
        loss_type_int: int,
        pos_qi: Tensor,
        pos_di: Tensor,
        num_pos: Tensor,
        has_neg: Tensor,
        pos_scores: Tensor,
        pos_label_values: Tensor,
    ):
        num_queries = q_scaled.shape[0]
        device = q_scaled.device

        if loss_type_int == LOSS_MULTI:
            neg_lse = _neg_lse_forward_sm100(q_scaled, k, labels)
            pos_neg_lse = neg_lse[pos_qi]

            valid = (num_pos > 0) & has_neg
            valid_f = valid.to(torch.float32)
            denom = valid_f.sum().clamp(min=1.0)
            num_pos_f = num_pos.clamp(min=1).to(torch.float32)

            per_pos_loss = F.softplus(pos_neg_lse - pos_scores)
            query_loss_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            query_loss_sum.scatter_add_(0, pos_qi, per_pos_loss)
            loss = ((query_loss_sum / num_pos_f) * valid_f).sum() / denom

            sigmoid_vals = torch.sigmoid(pos_neg_lse - pos_scores)
            aux = torch.zeros(num_queries, device=device, dtype=torch.float32)
            aux.scatter_add_(0, pos_qi, sigmoid_vals)
            ref_lse = neg_lse
            inv_weight = valid_f / (denom * num_pos_f)

        elif loss_type_int == LOSS_SOFT:
            all_lse = _all_lse_forward_sm100(q_scaled, k, labels)

            label_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            label_sum.scatter_add_(0, pos_qi, pos_label_values)
            valid = label_sum > 0
            valid_f = valid.to(torch.float32)
            denom = valid_f.sum().clamp(min=1.0)

            safe_label_sum = label_sum[pos_qi].clamp(min=1e-9)
            norm_labels = pos_label_values / safe_label_sum
            weighted_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            weighted_sum.scatter_add_(0, pos_qi, norm_labels * pos_scores)

            query_losses = all_lse - weighted_sum
            query_losses = torch.where(valid, query_losses, torch.zeros_like(query_losses))
            loss = query_losses.sum() / denom

            ref_lse = all_lse
            aux = label_sum
            inv_weight = valid_f / denom

        else:  # LOSS_CROSS
            all_lse = _all_lse_forward_sm100(q_scaled, k, labels)

            label_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            label_sum.scatter_add_(0, pos_qi, pos_label_values)
            valid = label_sum > 0
            valid_f = valid.to(torch.float32)
            denom = valid_f.sum().clamp(min=1.0)

            weighted_sum = torch.zeros(num_queries, device=device, dtype=torch.float32)
            weighted_sum.scatter_add_(0, pos_qi, pos_label_values * pos_scores)

            query_losses = label_sum * all_lse - weighted_sum
            query_losses = torch.where(valid, query_losses, torch.zeros_like(query_losses))
            loss = query_losses.sum() / denom

            ref_lse = all_lse
            aux = label_sum
            inv_weight = valid_f / denom

        ctx.save_for_backward(q_scaled, k, labels, ref_lse, aux, inv_weight)
        ctx.loss_type_int = loss_type_int
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        q_scaled, k, labels, ref_lse, aux, inv_weight = ctx.saved_tensors
        w = grad_output * inv_weight

        dq_scaled, dk = _backward_sm100(
            q_scaled, k, labels, ref_lse, aux, w,
            loss_type_int=ctx.loss_type_int,
        )

        dq_scaled = dq_scaled.to(q_scaled.dtype)
        dk = dk.to(k.dtype)
        return dq_scaled, dk, None, None, None, None, None, None, None, None


def fused_dense_loss_sm100(
    q: Tensor,
    k: Tensor,
    labels: Tensor,
    scale: "float | Tensor",
    loss_type: str = "multi",
    pos_qi: Optional[Tensor] = None,
    pos_di: Optional[Tensor] = None,
    pos_counts: Optional[Tensor] = None,
    neg_counts: Optional[Tensor] = None,
) -> Tensor:
    """
    SM100-optimized fused contrastive loss.

    Same interface as fused_dense_loss() in loss.py.
    Only activates on SM100+ GPUs. Falls back gracefully if extension
    fails to load.

    Args:
        q: [Q, D] query embeddings (normalized), bf16
        k: [K, D] document embeddings (normalized), bf16
        labels: [Q, K] int8 labels (>0: pos, 0: neg, -1: ignore)
        scale: temperature scale
        loss_type: "multi", "soft", "cross"
        pos_qi, pos_di, pos_counts, neg_counts: optional precomputed indices
    """
    loss_type_int = _LOSS_TYPE_MAP.get(loss_type, LOSS_MULTI)
    q_scaled = q * scale

    pos_qi, pos_di, num_pos, has_neg, pos_scores, pos_label_values = (
        _resolve_positive_pairs(q_scaled, k, labels, pos_qi, pos_di, pos_counts, neg_counts)
    )

    return FusedDenseLossSM100.apply(
        q_scaled, k, labels, loss_type_int,
        pos_qi, pos_di, num_pos, has_neg, pos_scores, pos_label_values,
    )
