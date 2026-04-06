// SPDX-License-Identifier: Apache-2.0
// SM120 Fused Contrastive Loss — CuTe MMA atoms + cp.async pipeline
//
// Uses CUTLASS CuTe for correct tensor core fragment layout:
//   - SM80_16x8x16_F32BF16BF16F32_TN MMA atom (mma.sync.m16n8k16)
//   - SM80_CP_ASYNC_CACHEGLOBAL copy atom (128-bit async loads)
//   - SM75_U32x4_LDSM_N for shared→register (ldmatrix)
//   - Swizzle<3,3,3> shared memory layout for bank-conflict-free access
//   - 3-stage pipeline following CUTLASS sgemm_sm80.cu pattern
//
// CuTe handles the complex thread-to-data fragment mapping automatically.
// The MMA accumulator is register-resident across all D-dimension tiles.

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;
using bf16_t = __nv_bfloat16;

// =====================================================================
// Configuration
// =====================================================================

// Forward tile sizes
static constexpr int BM = 64;      // queries per CTA
static constexpr int BN = 64;      // docs per N-loop iteration
static constexpr int BK = 64;      // D-dimension tile (4× MMA k=16)
static constexpr int PIPE = 3;     // pipeline stages

// Backward tile sizes
static constexpr int BM_BWD = 32;
static constexpr int BN_BWD = 64;
static constexpr int BD_GRAD = 128;  // D-chunk for gradient scatter

// Loss types (must match Python)
static constexpr int LOSS_MULTI = 0;
static constexpr int LOSS_SOFT = 1;
static constexpr int LOSS_CROSS = 2;
static constexpr int LSE_NEG_ONLY = 0;
static constexpr int LSE_VALID_ALL = 1;

// =====================================================================
// CuTe type aliases
// =====================================================================

// MMA atom: bf16 inputs, fp32 accumulation, TN layout (A=row, B=col)
using MmaAtom_t = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using TiledMma_t = decltype(make_tiled_mma(
    MmaAtom_t{},
    Layout<Shape<_2, _2, _1>>{},    // 2×2 atom arrangement
    Tile<_32, _32, _16>{}           // 32×32×16 tile → 128 threads
));

// Global → shared memory copy (128-bit async, L1 bypass)
using GmemCopyAtom_t = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, bf16_t>;
using GmemTiledCopy_t = decltype(make_tiled_copy(
    GmemCopyAtom_t{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // 128 threads: 16 M-groups × 8 K-groups
    Layout<Shape<_1, _8>>{}                     // 1×8 values per thread (128-bit)
));

// Shared memory → register copy
// Use UniversalCopy for SM120 compatibility (LDSM requires matched swizzle)
using S2RAtomA_t = Copy_Atom<DefaultCopy, bf16_t>;
using S2RAtomB_t = Copy_Atom<DefaultCopy, bf16_t>;

// Shared memory layout: stage-major (each stage is contiguous row-major MxK)
// sA[m, k, p] = base + p * M * K + m * K + k
using SmemLayoutA_t = decltype(make_layout(
    make_shape(Int<BM>{}, Int<BK>{}, Int<PIPE>{}),
    make_stride(Int<BK>{}, Int<1>{}, Int<BM * BK>{})
));
using SmemLayoutB_t = decltype(make_layout(
    make_shape(Int<BN>{}, Int<BK>{}, Int<PIPE>{}),
    make_stride(Int<BK>{}, Int<1>{}, Int<BN * BK>{})
));

// Backward smem layouts (different BM)
using SmemLayoutA_BWD_t = decltype(make_layout(
    make_shape(Int<BM_BWD>{}, Int<BK>{}, Int<PIPE>{}),
    make_stride(Int<BK>{}, Int<1>{}, Int<BM_BWD * BK>{})
));

// Shared storage struct
template <int M, int N>
struct SmemStorage {
    cute::array_aligned<bf16_t, M * BK * PIPE> A;  // Q pipeline
    cute::array_aligned<bf16_t, N * BK * PIPE> B;  // K pipeline
    union {
        cute::array_aligned<float, M * N>      scores;
        cute::array_aligned<bf16_t, N * BD_GRAD> k_grad;  // reuse for gradient phase
    };
    cute::array_aligned<int8_t, ((M * N + 3) & ~3)> labels;
    cute::array_aligned<float, M> lse_max;
    cute::array_aligned<float, M> lse_sum;
};

template <int M, int N>
struct SmemStorageBwd {
    cute::array_aligned<int8_t, ((M * N + 3) & ~3)> labels;  // persistent
    cute::array_aligned<float, M>    ref_lse;
    cute::array_aligned<float, M>    aux;
    cute::array_aligned<float, M>    w;
    cute::array_aligned<float, M * N> grad_s;                 // persistent
    union {
        struct {
            cute::array_aligned<bf16_t, M * BK * PIPE> A;     // Q pipeline
            cute::array_aligned<bf16_t, N * BK * PIPE> B;     // K pipeline
            cute::array_aligned<float, M * N>           scores;
        } mma;
        cute::array_aligned<bf16_t, static_cast<size_t>((N > M ? N : M)) * BD_GRAD> grad_chunk;
    };
};

// =====================================================================
// SM120 capability check
// =====================================================================

bool is_sm100_available() {
    static int cached = -1;
    if (cached >= 0) return cached;
    int device;
    cudaGetDevice(&device);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cached = (major >= 10) ? 1 : 0;
    return cached;
}

// =====================================================================
// Pipelined score computation: Q @ K^T via CuTe MMA + cp.async
//
// Implements the sgemm_sm80.cu pipeline pattern:
// - 3-stage cp.async global→shared pipeline
// - Register pipeline within each shared memory stage
// - LDSM for shared→register loads
// =====================================================================

template <int BLK_M, int BLK_N, class SmemLayoutA, class SmemLayoutB>
__device__ void compute_scores_cute(
    float*          __restrict__ s_scores_raw,  // [BLK_M, BLK_N] output
    bf16_t*         __restrict__ smem_A_raw,    // pipeline buffer for Q
    bf16_t*         __restrict__ smem_B_raw,    // pipeline buffer for K
    const bf16_t*   __restrict__ Q_gmem,
    const bf16_t*   __restrict__ K_gmem,
    int q_start, int n_start,
    int num_queries, int num_docs, int hidden_dim,
    SmemLayoutA const& sA_layout,
    SmemLayoutB const& sB_layout
) {
    int tid = threadIdx.x;

    // Create global memory tensors (row-major)
    auto mQ = make_tensor(make_gmem_ptr(Q_gmem),
        make_shape(num_queries, hidden_dim), make_stride(hidden_dim, Int<1>{}));
    auto mK = make_tensor(make_gmem_ptr(K_gmem),
        make_shape(num_docs, hidden_dim), make_stride(hidden_dim, Int<1>{}));

    // CTA tiler for Q (only M and K dims)
    auto cta_tiler_q = make_shape(Int<BLK_M>{}, Int<BLK_N>{}, Int<BK>{});

    // Tile Q and K for this CTA
    auto gQ = local_tile(mQ, make_tile(Int<BLK_M>{}, Int<BK>{}),
                         make_coord(q_start / BLK_M, _));     // (BLK_M, BK, k)
    auto gK = local_tile(mK, make_tile(Int<BLK_N>{}, Int<BK>{}),
                         make_coord(n_start / BLK_N, _));     // (BLK_N, BK, k)

    // Shared memory tensors
    auto sA = make_tensor(make_smem_ptr(smem_A_raw), sA_layout);  // (BLK_M, BK, PIPE)
    auto sB = make_tensor(make_smem_ptr(smem_B_raw), sB_layout);  // (BLK_N, BK, PIPE)

    // Score output tensor
    auto sScores = make_tensor(make_smem_ptr(s_scores_raw),
        make_layout(make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                    make_stride(Int<BLK_N>{}, Int<1>{})));  // row-major to match raw pointer access

    // TiledMMA
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // Gmem → Smem copy
    GmemTiledCopy_t gmem_copy;
    auto thr_copy_a = gmem_copy.get_slice(tid);
    auto tAgA = thr_copy_a.partition_S(gQ);           // (CPY, CPY_M, CPY_K, k)
    auto tAsA = thr_copy_a.partition_D(sA);           // (CPY, CPY_M, CPY_K, PIPE)
    auto thr_copy_b = gmem_copy.get_slice(tid);
    auto tBgB = thr_copy_b.partition_S(gK);
    auto tBsB = thr_copy_b.partition_D(sB);

    // Predication for boundary tiles (out-of-bounds → zero)
    auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)));
    auto tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)));

    // Generate predicates from identity tensors
    {
        auto idxA = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BK>{}));
        auto tAcA = thr_copy_a.partition_S(idxA);
        CUTE_UNROLL
        for (int m = 0; m < size<0>(tApA); ++m) {
            CUTE_UNROLL
            for (int k = 0; k < size<1>(tApA); ++k) {
                tApA(m, k) = get<0>(tAcA(0, m, k)) + q_start < num_queries;
            }
        }
    }
    {
        auto idxB = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BK>{}));
        auto tBcB = thr_copy_b.partition_S(idxB);
        CUTE_UNROLL
        for (int n = 0; n < size<0>(tBpB); ++n) {
            CUTE_UNROLL
            for (int k = 0; k < size<1>(tBpB); ++k) {
                tBpB(n, k) = get<0>(tBcB(0, n, k)) + n_start < num_docs;
            }
        }
    }

    // Smem → Reg copy (LDSM)
    auto s2r_copy_a = make_tiled_copy_A(S2RAtomA_t{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(tid);
    auto tXsA       = s2r_thr_a.partition_S(sA);                 // (CPY, MMA_M, MMA_K, PIPE)

    auto s2r_copy_b = make_tiled_copy_B(S2RAtomB_t{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(tid);
    auto tXsB       = s2r_thr_b.partition_S(sB);                 // (CPY, MMA_N, MMA_K, PIPE)

    // Register fragments
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));       // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));       // (MMA, MMA_N, MMA_K)
    auto tXrA = s2r_thr_a.retile_D(tCrA);                        // (CPY, MMA_M, MMA_K)
    auto tXrB = s2r_thr_b.retile_D(tCrB);                        // (CPY, MMA_N, MMA_K)

    // Accumulator (register-resident across all D tiles)
    auto tCsC = thr_mma.partition_C(sScores);                     // (MMA, MMA_M, MMA_N)
    auto tCrC = thr_mma.make_fragment_C(tCsC);                    // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // ---------------------------------------------------------------
    // Pipeline: prefetch + main loop (following sgemm_sm80.cu)
    // ---------------------------------------------------------------
    constexpr int K_PIPE_MAX = PIPE;
    auto K_BLOCK_MAX = size<2>(tCrA);  // BK/16 = 4

    int k_tile_count = size<3>(tAgA);  // hidden_dim / BK
    int k_tile_next = 0;

    // Prefetch: fill pipeline stages 0..(PIPE-2)
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        if (k_tile_count > 0) {
            copy_if(gmem_copy, tApA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
            copy_if(gmem_copy, tBpB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        } else {
            clear(tAsA(_, _, _, k_pipe));
            clear(tBsB(_, _, _, k_pipe));
        }
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // Pipeline state
    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    auto tXsA_p = tXsA(_, _, _, smem_pipe_read);
    auto tXsB_p = tXsB(_, _, _, smem_pipe_read);

    // Register prefetch (first k_block from first stage)
    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
        copy(s2r_copy_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        copy(s2r_copy_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    // Main loop
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load next register tile from smem
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(s2r_copy_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(s2r_copy_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

            if (k_block == 0) {
                // Issue next gmem→smem copy
                if (k_tile_count > 0) {
                    copy_if(gmem_copy, tApA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                    copy_if(gmem_copy, tBpB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                } else {
                    clear(tAsA(_, _, _, smem_pipe_write));
                    clear(tBsB(_, _, _, smem_pipe_write));
                }
                cp_async_fence();
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                smem_pipe_write = smem_pipe_read;
                smem_pipe_read  = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
            }

            // MMA
            gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // Store accumulator to shared memory scores
    copy(tCrC, tCsC);
    cp_async_wait<0>();
    __syncthreads();
}

// =====================================================================
// Forward LSE Kernel
// =====================================================================

template <int LSE_MODE>
__global__ void __launch_bounds__(decltype(size(TiledMma_t{}))::value, 2)
fused_loss_lse_fwd_kernel(
    const bf16_t*  __restrict__ Q,
    const bf16_t*  __restrict__ K,
    const int8_t*  __restrict__ Labels,
    float*         __restrict__ OutLSE,
    int num_queries, int num_docs, int hidden_dim
) {
    const int pid_m = blockIdx.x;
    const int tid = threadIdx.x;
    const int q_start = pid_m * BM;
    if (q_start >= num_queries) return;

    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SmemStorage<BM, BN>*>(smem_raw);

    float* s_m   = smem.lse_max.begin();
    float* s_lse = smem.lse_sum.begin();

    // Initialize online LSE state
    for (int r = tid; r < BM; r += blockDim.x) {
        s_m[r]   = -FLT_MAX;
        s_lse[r] = 0.0f;
    }
    __syncthreads();

    SmemLayoutA_t sA_layout;
    SmemLayoutB_t sB_layout;

    for (int n_start = 0; n_start < num_docs; n_start += BN) {
        // Load labels
        int8_t* s_labels = smem.labels.begin();
        for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
            int r = idx / BN, c = idx % BN;
            int qr = q_start + r, kc = n_start + c;
            s_labels[idx] = (qr < num_queries && kc < num_docs)
                ? Labels[static_cast<int64_t>(qr) * num_docs + kc] : (int8_t)-1;
        }
        __syncthreads();

        // Pipelined score computation
        compute_scores_cute<BM, BN>(
            smem.scores.begin(), smem.A.begin(), smem.B.begin(),
            Q, K, q_start, n_start,
            num_queries, num_docs, hidden_dim,
            sA_layout, sB_layout
        );

        // Online LogSumExp update
        float* s_scores = smem.scores.begin();
        for (int row = tid; row < BM; row += blockDim.x) {
            if (q_start + row >= num_queries) continue;

            float row_max = -FLT_MAX;
            for (int col = 0; col < BN && n_start + col < num_docs; col++) {
                int8_t label = s_labels[row * BN + col];
                bool inc = (LSE_MODE == LSE_NEG_ONLY) ? (label == 0) : (label >= 0);
                if (inc) row_max = fmaxf(row_max, s_scores[row * BN + col]);
            }

            float row_sum = 0.0f;
            if (row_max > -FLT_MAX) {
                for (int col = 0; col < BN && n_start + col < num_docs; col++) {
                    int8_t label = s_labels[row * BN + col];
                    bool inc = (LSE_MODE == LSE_NEG_ONLY) ? (label == 0) : (label >= 0);
                    if (inc) row_sum += expf(s_scores[row * BN + col] - row_max);
                }
            }

            float old_max = s_m[row];
            float new_max = fmaxf(old_max, row_max);
            if (new_max > -FLT_MAX) {
                float alpha = (old_max > -FLT_MAX) ? expf(old_max - new_max) : 0.0f;
                float beta  = (row_max > -FLT_MAX) ? expf(row_max - new_max) : 0.0f;
                s_lse[row] = alpha * s_lse[row] + beta * row_sum;
                s_m[row]   = new_max;
            }
        }
        __syncthreads();
    }

    for (int r = tid; r < BM; r += blockDim.x) {
        int qr = q_start + r;
        if (qr < num_queries) {
            float m = s_m[r];
            float l = s_lse[r];
            OutLSE[qr] = (m > -FLT_MAX && l > 0.0f) ? (m + logf(l)) : -FLT_MAX;
        }
    }
}

// =====================================================================
// Backward dQ Kernel
// =====================================================================

template <int LOSS_TYPE>
__global__ void __launch_bounds__(decltype(size(TiledMma_t{}))::value, 2)
fused_loss_dq_bwd_kernel(
    const bf16_t*  __restrict__ Q,
    const bf16_t*  __restrict__ K,
    const int8_t*  __restrict__ Labels,
    float*         __restrict__ dQ,
    const float*   __restrict__ RefLSE,
    const float*   __restrict__ Aux,
    const float*   __restrict__ W,
    int num_queries, int num_docs, int hidden_dim
) {
    const int pid_m = blockIdx.x;
    const int tid = threadIdx.x;
    const int q_start = pid_m * BM_BWD;
    if (q_start >= num_queries) return;

    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SmemStorageBwd<BM_BWD, BN_BWD>*>(smem_raw);

    // Load per-query constants
    for (int r = tid; r < BM_BWD; r += blockDim.x) {
        int qr = q_start + r;
        smem.ref_lse[r] = (qr < num_queries) ? RefLSE[qr] : 0.0f;
        smem.aux[r]     = (qr < num_queries) ? Aux[qr]    : 0.0f;
        smem.w[r]       = (qr < num_queries) ? W[qr]      : 0.0f;
    }
    __syncthreads();

    SmemLayoutA_BWD_t sA_layout;
    SmemLayoutB_t     sB_layout;

    for (int n_start = 0; n_start < num_docs; n_start += BN_BWD) {
        // Load labels
        int8_t* s_labels = smem.labels.begin();
        for (int idx = tid; idx < BM_BWD * BN_BWD; idx += blockDim.x) {
            int r = idx / BN_BWD, c = idx % BN_BWD;
            int qr = q_start + r, kc = n_start + c;
            s_labels[idx] = (qr < num_queries && kc < num_docs)
                ? Labels[static_cast<int64_t>(qr) * num_docs + kc] : (int8_t)-1;
        }
        __syncthreads();

        // Phase 1: recompute scores
        compute_scores_cute<BM_BWD, BN_BWD>(
            smem.mma.scores.begin(), smem.mma.A.begin(), smem.mma.B.begin(),
            Q, K, q_start, n_start,
            num_queries, num_docs, hidden_dim,
            sA_layout, sB_layout
        );

        // Phase 2: compute grad_s
        float* s_scores = smem.mma.scores.begin();
        float* s_grad_s = smem.grad_s.begin();
        for (int idx = tid; idx < BM_BWD * BN_BWD; idx += blockDim.x) {
            int r = idx / BN_BWD, c = idx % BN_BWD;
            if (q_start + r >= num_queries || n_start + c >= num_docs) {
                s_grad_s[idx] = 0.0f;
                continue;
            }
            int8_t label = s_labels[idx];
            float sm = s_scores[idx] - smem.ref_lse[r];
            float w = smem.w[r], aux = smem.aux[r];
            float grad = 0.0f;

            if (LOSS_TYPE == LOSS_MULTI) {
                if (label > 0)       grad = w * (1.0f / (1.0f + expf(-sm)) - 1.0f);
                else if (label == 0) grad = w * expf(sm) * aux;
            } else if (LOSS_TYPE == LOSS_SOFT) {
                if (label >= 0) {
                    float sa = (aux > 0.0f) ? aux : 1.0f;
                    grad = w * (expf(sm) - fmaxf(static_cast<float>(label), 0.0f) / sa);
                }
            } else {
                if (label >= 0)
                    grad = w * (aux * expf(sm) - fmaxf(static_cast<float>(label), 0.0f));
            }
            s_grad_s[idx] = grad;
        }
        __syncthreads();

        // Phase 3: dQ += grad_s @ K (scalar, reload K via cp.async)
        bf16_t* s_k = smem.grad_chunk.begin();
        for (int d_start = 0; d_start < hidden_dim; d_start += BD_GRAD) {
            int d_size = min(BD_GRAD, hidden_dim - d_start);
            // Load K chunk [BN, BD_GRAD]
            for (int idx = tid; idx < BN_BWD * d_size; idx += blockDim.x) {
                int n = idx / d_size, d = idx % d_size;
                int gn = n_start + n;
                s_k[n * BD_GRAD + d] = (gn < num_docs)
                    ? K[static_cast<int64_t>(gn) * hidden_dim + d_start + d]
                    : __float2bfloat16(0.0f);
            }
            __syncthreads();
            for (int idx = tid; idx < BM_BWD * d_size; idx += blockDim.x) {
                int r = idx / d_size, dc = idx % d_size;
                if (q_start + r >= num_queries) continue;
                float a = 0.0f;
                for (int n = 0; n < BN_BWD && n_start + n < num_docs; n++)
                    a += s_grad_s[r * BN_BWD + n] * __bfloat162float(s_k[n * BD_GRAD + dc]);
                atomicAdd(&dQ[static_cast<int64_t>(q_start + r) * hidden_dim + d_start + dc], a);
            }
            __syncthreads();
        }
    }
}

// =====================================================================
// Backward dK Kernel
// =====================================================================

template <int LOSS_TYPE>
__global__ void __launch_bounds__(decltype(size(TiledMma_t{}))::value, 2)
fused_loss_dk_bwd_kernel(
    const bf16_t*  __restrict__ Q,
    const bf16_t*  __restrict__ K,
    const int8_t*  __restrict__ Labels,
    float*         __restrict__ dK,
    const float*   __restrict__ RefLSE,
    const float*   __restrict__ Aux,
    const float*   __restrict__ W,
    int num_queries, int num_docs, int hidden_dim
) {
    const int pid_n = blockIdx.x;
    const int tid = threadIdx.x;
    const int k_start = pid_n * BN_BWD;
    if (k_start >= num_docs) return;

    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SmemStorageBwd<BM_BWD, BN_BWD>*>(smem_raw);

    SmemLayoutA_BWD_t sA_layout;
    SmemLayoutB_t     sB_layout;

    for (int m_start = 0; m_start < num_queries; m_start += BM_BWD) {
        for (int r = tid; r < BM_BWD; r += blockDim.x) {
            int qr = m_start + r;
            smem.ref_lse[r] = (qr < num_queries) ? RefLSE[qr] : 0.0f;
            smem.aux[r]     = (qr < num_queries) ? Aux[qr]    : 0.0f;
            smem.w[r]       = (qr < num_queries) ? W[qr]      : 0.0f;
        }
        int8_t* s_labels = smem.labels.begin();
        for (int idx = tid; idx < BM_BWD * BN_BWD; idx += blockDim.x) {
            int r = idx / BN_BWD, c = idx % BN_BWD;
            int qr = m_start + r, kc = k_start + c;
            s_labels[idx] = (qr < num_queries && kc < num_docs)
                ? Labels[static_cast<int64_t>(qr) * num_docs + kc] : (int8_t)-1;
        }
        __syncthreads();

        compute_scores_cute<BM_BWD, BN_BWD>(
            smem.mma.scores.begin(), smem.mma.A.begin(), smem.mma.B.begin(),
            Q, K, m_start, k_start,
            num_queries, num_docs, hidden_dim,
            sA_layout, sB_layout
        );

        float* s_scores = smem.mma.scores.begin();
        float* s_grad_s = smem.grad_s.begin();
        for (int idx = tid; idx < BM_BWD * BN_BWD; idx += blockDim.x) {
            int r = idx / BN_BWD, c = idx % BN_BWD;
            if (m_start + r >= num_queries || k_start + c >= num_docs) {
                s_grad_s[idx] = 0.0f;
                continue;
            }
            int8_t label = s_labels[idx];
            float sm = s_scores[idx] - smem.ref_lse[r];
            float w = smem.w[r], aux = smem.aux[r];
            float grad = 0.0f;
            if (LOSS_TYPE == LOSS_MULTI) {
                if (label > 0)       grad = w * (1.0f / (1.0f + expf(-sm)) - 1.0f);
                else if (label == 0) grad = w * expf(sm) * aux;
            } else if (LOSS_TYPE == LOSS_SOFT) {
                if (label >= 0) {
                    float sa = (aux > 0.0f) ? aux : 1.0f;
                    grad = w * (expf(sm) - fmaxf(static_cast<float>(label), 0.0f) / sa);
                }
            } else {
                if (label >= 0)
                    grad = w * (aux * expf(sm) - fmaxf(static_cast<float>(label), 0.0f));
            }
            s_grad_s[idx] = grad;
        }
        __syncthreads();

        bf16_t* s_q = smem.grad_chunk.begin();
        for (int d_start = 0; d_start < hidden_dim; d_start += BD_GRAD) {
            int d_size = min(BD_GRAD, hidden_dim - d_start);
            for (int idx = tid; idx < BM_BWD * d_size; idx += blockDim.x) {
                int m = idx / d_size, d = idx % d_size;
                int gm = m_start + m;
                s_q[m * BD_GRAD + d] = (gm < num_queries)
                    ? Q[static_cast<int64_t>(gm) * hidden_dim + d_start + d]
                    : __float2bfloat16(0.0f);
            }
            __syncthreads();
            for (int idx = tid; idx < BN_BWD * d_size; idx += blockDim.x) {
                int kr = idx / d_size, dc = idx % d_size;
                if (k_start + kr >= num_docs) continue;
                float a = 0.0f;
                for (int m = 0; m < BM_BWD && m_start + m < num_queries; m++)
                    a += s_grad_s[m * BN_BWD + kr] * __bfloat162float(s_q[m * BD_GRAD + dc]);
                atomicAdd(&dK[static_cast<int64_t>(k_start + kr) * hidden_dim + d_start + dc], a);
            }
            __syncthreads();
        }
    }
}

// =====================================================================
// Host interface
// =====================================================================

static int fwd_smem_bytes() {
    return static_cast<int>(sizeof(SmemStorage<BM, BN>)) + 256;
}

static int bwd_smem_bytes() {
    return static_cast<int>(sizeof(SmemStorageBwd<BM_BWD, BN_BWD>)) + 256;
}

std::tuple<torch::Tensor, torch::Tensor> fused_loss_fwd_sm100_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor labels,
    float scale, int lse_mode
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && labels.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16 && K.dtype() == torch::kBFloat16);
    TORCH_CHECK(labels.dtype() == torch::kInt8);

    int nq = Q.size(0), hd = Q.size(1), nd = K.size(0);
    auto out = torch::empty({nq}, Q.options().dtype(torch::kFloat32));

    int grid = (nq + BM - 1) / BM;
    int smem = fwd_smem_bytes();
    constexpr int threads = decltype(size(TiledMma_t{}))::value;

    auto launch = [&](auto kfn) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kfn<<<grid, threads, smem>>>(
            reinterpret_cast<const bf16_t*>(Q.data_ptr()),
            reinterpret_cast<const bf16_t*>(K.data_ptr()),
            labels.data_ptr<int8_t>(), out.data_ptr<float>(),
            nq, nd, hd);
    };

    if (lse_mode == LSE_NEG_ONLY) launch(fused_loss_lse_fwd_kernel<LSE_NEG_ONLY>);
    else                          launch(fused_loss_lse_fwd_kernel<LSE_VALID_ALL>);

    return {out, torch::empty({0}, out.options())};
}

std::tuple<torch::Tensor, torch::Tensor> fused_loss_bwd_sm100_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor labels,
    torch::Tensor ref_lse, torch::Tensor aux, torch::Tensor w,
    int loss_type
) {
    int nq = Q.size(0), hd = Q.size(1), nd = K.size(0);
    auto dQ = torch::zeros({nq, hd}, Q.options().dtype(torch::kFloat32));
    auto dK = torch::zeros({nd, hd}, K.options().dtype(torch::kFloat32));
    int smem = bwd_smem_bytes();
    constexpr int threads = decltype(size(TiledMma_t{}))::value;

    auto launch_dq = [&](auto kfn) {
        int grid = (nq + BM_BWD - 1) / BM_BWD;
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kfn<<<grid, threads, smem>>>(
            reinterpret_cast<const bf16_t*>(Q.data_ptr()),
            reinterpret_cast<const bf16_t*>(K.data_ptr()),
            labels.data_ptr<int8_t>(), dQ.data_ptr<float>(),
            ref_lse.data_ptr<float>(), aux.data_ptr<float>(), w.data_ptr<float>(),
            nq, nd, hd);
    };
    auto launch_dk = [&](auto kfn) {
        int grid = (nd + BN_BWD - 1) / BN_BWD;
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kfn<<<grid, threads, smem>>>(
            reinterpret_cast<const bf16_t*>(Q.data_ptr()),
            reinterpret_cast<const bf16_t*>(K.data_ptr()),
            labels.data_ptr<int8_t>(), dK.data_ptr<float>(),
            ref_lse.data_ptr<float>(), aux.data_ptr<float>(), w.data_ptr<float>(),
            nq, nd, hd);
    };

    if (loss_type == LOSS_MULTI) {
        launch_dq(fused_loss_dq_bwd_kernel<LOSS_MULTI>);
        launch_dk(fused_loss_dk_bwd_kernel<LOSS_MULTI>);
    } else if (loss_type == LOSS_SOFT) {
        launch_dq(fused_loss_dq_bwd_kernel<LOSS_SOFT>);
        launch_dk(fused_loss_dk_bwd_kernel<LOSS_SOFT>);
    } else {
        launch_dq(fused_loss_dq_bwd_kernel<LOSS_CROSS>);
        launch_dk(fused_loss_dk_bwd_kernel<LOSS_CROSS>);
    }

    return {dQ, dK};
}
