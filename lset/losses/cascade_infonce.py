"""Cascade InfoNCE: proxy-based candidate selection + truncated loss.

Replaces the full B×B similarity matrix with:
1. Tiled proxy topk (first d_small dims, no B² matrix)
2. Full-dim similarity for candidates only (chunked gather+bmm)
3. Cross-entropy over candidates
"""

import torch
import torch.nn.functional as F


def _tiled_proxy_topk(q_proxy, k_proxy, K_prime, tile_size=1024, qchunk=256):
    """Query-chunked tiled proxy topk. Memory: O(qchunk×tile + B×K')."""
    B_q = q_proxy.shape[0]
    B_k = k_proxy.shape[0]
    device = q_proxy.device

    topk_vals = torch.full((B_q, K_prime), float("-inf"), device=device)
    topk_idx = torch.zeros((B_q, K_prime), dtype=torch.int32, device=device)

    for q_start in range(0, B_q, qchunk):
        q_end = min(q_start + qchunk, B_q)
        q_chunk = q_proxy[q_start:q_end]
        qc = q_end - q_start

        qc_vals = topk_vals[q_start:q_end].clone()
        qc_idx = topk_idx[q_start:q_end].clone()
        merge_vals = torch.empty(qc, 2 * K_prime, device=device)
        merge_idx = torch.empty(qc, 2 * K_prime, dtype=torch.int32, device=device)

        for j_start in range(0, B_k, tile_size):
            j_end = min(j_start + tile_size, B_k)
            actual = j_end - j_start

            S_tile = q_chunk @ k_proxy[j_start:j_end].T  # (qc, actual)

            # Mask diagonal (self-similarity) — only if q and k are same set
            if B_q == B_k:
                for qi in range(q_start, q_end):
                    if j_start <= qi < j_end:
                        S_tile[qi - q_start, qi - j_start] = float("-inf")

            tk = min(K_prime, actual)
            tile_tv, tile_tl = S_tile[:, :actual].topk(tk, dim=1)
            tile_ti = (tile_tl + j_start).to(torch.int32)

            merge_vals[:, :K_prime].copy_(qc_vals)
            merge_vals[:, K_prime:K_prime + tk].copy_(tile_tv)
            merge_idx[:, :K_prime].copy_(qc_idx)
            merge_idx[:, K_prime:K_prime + tk].copy_(tile_ti)

            _, sel = merge_vals[:, :K_prime + tk].topk(K_prime, dim=1)
            torch.gather(merge_vals[:, :K_prime + tk], 1, sel, out=qc_vals)
            torch.gather(merge_idx[:, :K_prime + tk], 1, sel.int(), out=qc_idx)

        topk_vals[q_start:q_end] = qc_vals
        topk_idx[q_start:q_end] = qc_idx

    return topk_idx.long()


def cascade_infonce_loss(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    temperature: float = 0.02,
    d_small: int = 64,
    K_prime: int = 256,
    tile_size: int = 1024,
    qchunk: int = 256,
    refine_chunk: int = 64,
) -> torch.Tensor:
    """Cascade InfoNCE loss without materializing B×B similarity matrix.

    Args:
        query_embeds: (B, d) L2-normalized query embeddings.
        doc_embeds: (B, d) L2-normalized doc embeddings.
        temperature: Scaling temperature.
        d_small: Proxy dimension (first d_small dims).
        K_prime: Candidates per query from proxy.
        tile_size: Key tile size for proxy phase.
        qchunk: Query chunk size for proxy phase.
        refine_chunk: Query chunk size for refine phase.

    Returns:
        Scalar loss.
    """
    B, d_val = query_embeds.shape
    device = query_embeds.device

    # Step 1: Proxy candidate selection (no grad)
    with torch.no_grad():
        q_proxy = F.normalize(query_embeds[:, :d_small], dim=1)
        k_proxy = F.normalize(doc_embeds[:, :d_small], dim=1)
        cand_idx = _tiled_proxy_topk(q_proxy, k_proxy, K_prime, tile_size, qchunk)

        # Force-include positive (diagonal) in candidates
        diag_idx = torch.arange(B, device=device)
        pos_in_cand = (cand_idx == diag_idx.unsqueeze(1)).any(dim=1)  # (B,)
        missing = ~pos_in_cand
        if missing.any():
            cand_idx[missing, -1] = diag_idx[missing]

    # Step 2+3: Full-dim similarity + loss (chunked, WITH grad)
    total_loss = torch.tensor(0.0, device=device)

    for c in range(0, B, refine_chunk):
        c_end = min(c + refine_chunk, B)
        cs = c_end - c

        chunk_q = query_embeds[c:c_end]       # (cs, d)
        chunk_cand = cand_idx[c:c_end]         # (cs, K')

        # Gather candidate keys
        K_cand = doc_embeds[chunk_cand.reshape(-1)].reshape(cs, K_prime, d_val)  # (cs, K', d)
        S_cand = torch.bmm(
            chunk_q.unsqueeze(1), K_cand.transpose(1, 2)
        ).squeeze(1) / temperature  # (cs, K')

        # Labels: position of positive (diagonal) in candidates
        chunk_diag = torch.arange(c, c_end, device=device)
        pos_mask = (chunk_cand == chunk_diag.unsqueeze(1))
        labels = pos_mask.nonzero(as_tuple=True)[1]  # (cs,)

        total_loss = total_loss + F.cross_entropy(S_cand, labels, reduction="sum")

        del K_cand, S_cand

    return total_loss / B
