/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * FMHA Forward for GQA.
 *
 * Two implementations with the same interface:
 *
 *   1. Custom portable kernel (default, always available):
 *      Uses shared-memory tree reductions that work correctly on both wave32
 *      (RDNA3/GFX11) and wave64 (CDNA/GFX9). Supports causal masking for
 *      prefill and no-mask decode. Highest accuracy on RDNA3.
 *
 *   2. CK Tile naive_attention_fwd (optional, requires CK_ROOT at build time):
 *      Uses CK Tile's reference naive_attention_fwd host API. Its internal
 *      kernel uses wave64-hardcoded reductions (ds_bpermute with 6 stages,
 *      cross_wave_reduce with wave_size=64). On RDNA3 (wave32), the
 *      cross_wave_reduce only combines 4 of 8 wave32 groups, causing
 *      incomplete max/sum reductions and reduced accuracy.
 *
 * Build configuration:
 *   -DHAS_CK_TILE=1   Enables CK Tile naive_attention_fwd (set by CMake when CK_ROOT is found)
 *   (default)          Only portable kernel is available
 *
 * Runtime environment variable (only effective when built with HAS_CK_TILE):
 *   FMHA_USE_CK_NAIVE=1   Use CK Tile naive_attention_fwd
 *   (default)              Use custom portable kernel
 *
 * BHSD layout [Batch, Head, Seqlen, Hdim], FP16, GQA support.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cstdlib>

#ifdef HAS_CK_TILE
// CK Tile headers (only included when CK_ROOT is available)
#include "ck_tile/core.hpp"
#include "ck_tile/ref/naive_attention.hpp"
using DataType = ck_tile::half_t;
#else
// Without CK Tile, use HIP's native __half
using DataType = __half;
#endif

using AccType = float;

// Flag reflecting whether CK Tile is available at build time
#ifdef HAS_CK_TILE
static const bool kHasCkTile = true;
#else
static const bool kHasCkTile = false;
#endif

// =============================================================================
// Custom portable kernel (wave32 + wave64 safe) — always available
// =============================================================================
static constexpr int kBlockSize = 256;

struct FmhaCausalArgs
{
    const DataType* q_ptr;
    const DataType* k_ptr;
    const DataType* v_ptr;
    DataType*       o_ptr;
    int seqlen_q;
    int seqlen_k;
    int hdim;
    int nhead_q;
    int nhead_k;
    float scale;
};

// Portable shared-memory tree reductions (wave32 + wave64 safe)
__device__ __forceinline__
AccType block_reduce_max(AccType val, AccType* smem)
{
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    AccType result = smem[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__
AccType block_reduce_sum(AccType val, AccType* smem)
{
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    AccType result = smem[0];
    __syncthreads();
    return result;
}

// =============================================================================
// FMHA kernel with bottom-right causal masking
//
// Grid:  (ceil(hdim / kBlockSize), seqlen_q, batch * nhead_q)
// Block: (kBlockSize)
//
// Causal mask (bottom-right):
//   For prefill (seqlen_q == seqlen_k): standard causal, Q[i] attends K[0..i]
//   For decode  (seqlen_q == 1):        causal_limit = seqlen_k - 1, all keys
// =============================================================================
__global__ void __launch_bounds__(kBlockSize)
fmha_naive_causal_kernel(FmhaCausalArgs args)
{
    __shared__ union {
        AccType  reduce[kBlockSize];
        DataType p_vals[kBlockSize];
    } smem;

    const int i_dv         = blockIdx.x * kBlockSize + threadIdx.x;
    const int i_sq         = blockIdx.y;
    const int i_batch_head = blockIdx.z;
    const int i_bq         = i_batch_head / args.nhead_q;
    const int i_hq         = i_batch_head % args.nhead_q;
    const int i_hk         = i_hq / (args.nhead_q / args.nhead_k);

    const DataType* q_base = args.q_ptr
        + static_cast<int64_t>(i_bq) * args.nhead_q * args.seqlen_q * args.hdim
        + static_cast<int64_t>(i_hq) * args.seqlen_q * args.hdim;
    const DataType* k_base = args.k_ptr
        + static_cast<int64_t>(i_bq) * args.nhead_k * args.seqlen_k * args.hdim
        + static_cast<int64_t>(i_hk) * args.seqlen_k * args.hdim;
    const DataType* v_base = args.v_ptr
        + static_cast<int64_t>(i_bq) * args.nhead_k * args.seqlen_k * args.hdim
        + static_cast<int64_t>(i_hk) * args.seqlen_k * args.hdim;
    DataType* o_base = args.o_ptr
        + static_cast<int64_t>(i_bq) * args.nhead_q * args.seqlen_q * args.hdim
        + static_cast<int64_t>(i_hq) * args.seqlen_q * args.hdim;

    const DataType* q_row = q_base + static_cast<int64_t>(i_sq) * args.hdim;
    const int causal_limit = i_sq + (args.seqlen_k - args.seqlen_q);

    AccType row_max = -FLT_MAX;
    AccType l       = 0.0f;
    AccType o_acc   = 0.0f;

    const int sk_loops = (args.seqlen_k + kBlockSize - 1) / kBlockSize;

    for (int i_loop = 0; i_loop < sk_loops; i_loop++)
    {
        const int i_sk = i_loop * kBlockSize + threadIdx.x;

        AccType s_val = -FLT_MAX;
        if (i_sk < args.seqlen_k && i_sk <= causal_limit)
        {
            const DataType* k_row = k_base + static_cast<int64_t>(i_sk) * args.hdim;
            AccType dot = 0.0f;
            for (int d = 0; d < args.hdim; d++)
                dot += static_cast<AccType>(q_row[d]) * static_cast<AccType>(k_row[d]);
            s_val = dot * args.scale;
        }

        const AccType old_max = row_max;
        const AccType cur_max = block_reduce_max(s_val, smem.reduce);
        row_max = fmaxf(old_max, cur_max);

        const AccType p_val = expf(s_val - row_max);
        const AccType row_sum = block_reduce_sum(p_val, smem.reduce);

        const AccType rescale = expf(old_max - row_max);
        l     = rescale * l + row_sum;
        o_acc = o_acc * rescale;

        smem.p_vals[threadIdx.x] = static_cast<DataType>(p_val);
        __syncthreads();

        if (i_dv < args.hdim)
        {
            const int sk_start = i_loop * kBlockSize;
            for (int j = 0; j < kBlockSize; j++)
            {
                const int i_sv = sk_start + j;
                if (i_sv < args.seqlen_k)
                {
                    AccType p = static_cast<AccType>(smem.p_vals[j]);
                    AccType v = static_cast<AccType>(v_base[static_cast<int64_t>(i_sv) * args.hdim + i_dv]);
                    o_acc += p * v;
                }
            }
        }
        __syncthreads();
    }

    if (l > 0.0f)
        o_acc = o_acc / l;
    else
        o_acc = 0.0f;

    if (i_dv < args.hdim)
        o_base[static_cast<int64_t>(i_sq) * args.hdim + i_dv] = static_cast<DataType>(o_acc);
}

// =============================================================================
// Implementation 1: Custom portable kernel (default, always available)
// =============================================================================
static hipError_t launch_fmha_portable(
    hipStream_t stream,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    int32_t batch,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t hdim,
    float scale)
{
    FmhaCausalArgs args;
    args.q_ptr    = reinterpret_cast<const DataType*>(q_ptr);
    args.k_ptr    = reinterpret_cast<const DataType*>(k_ptr);
    args.v_ptr    = reinterpret_cast<const DataType*>(v_ptr);
    args.o_ptr    = reinterpret_cast<DataType*>(o_ptr);
    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.hdim     = hdim;
    args.nhead_q  = nhead_q;
    args.nhead_k  = nhead_k;
    args.scale    = scale;

    dim3 grid(
        (hdim + kBlockSize - 1) / kBlockSize,
        seqlen_q,
        batch * nhead_q
    );
    dim3 block(kBlockSize);

    const char* mode = (seqlen_q == 1) ? "Decode" : "Prefill";
    fprintf(stderr, "[Portable %s] batch=%d, nhead_q=%d, nhead_k=%d, "
            "seqlen_q=%d, seqlen_k=%d, hdim=%d, scale=%f, grid=(%d,%d,%d)\n",
            mode, batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim, scale,
            grid.x, grid.y, grid.z);

    hipLaunchKernelGGL(
        fmha_naive_causal_kernel,
        grid, block, 0, stream, args);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[Portable FMHA] Launch error: %s\n",
                hipGetErrorString(err));
        return err;
    }
    return hipSuccess;
}

// =============================================================================
// Implementation 2: CK Tile naive_attention_fwd (only when built with CK_ROOT)
// NOTE: Uses wave64-hardcoded reductions. On RDNA3 (wave32), the
// cross_wave_reduce only combines 4 of 8 wave groups, causing reduced accuracy.
// =============================================================================
#ifdef HAS_CK_TILE

static hipError_t launch_fmha_ck_naive(
    hipStream_t stream,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    int32_t batch,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t hdim,
    float scale)
{
    ck_tile::naive_attention_fwd_traits traits;
    traits.q_type      = "fp16";
    traits.k_type      = "fp16";
    traits.v_type      = "fp16";
    traits.o_type      = "fp16";
    traits.q_layout    = "bhsd";
    traits.k_layout    = "bhsd";
    traits.v_layout    = "bhsd";
    traits.o_layout    = "bhsd";
    traits.variation    = static_cast<int>(ck_tile::naive_attention_variation_enum::FLASH_BATCHED);
    traits.quant_algo   = static_cast<int>(ck_tile::naive_attention_quant_algo::NO);

    ck_tile::naive_attention_fwd_args ck_args;
    ck_args.q_ptr             = const_cast<void*>(q_ptr);
    ck_args.k_ptr             = const_cast<void*>(k_ptr);
    ck_args.v_ptr             = const_cast<void*>(v_ptr);
    ck_args.o_ptr             = o_ptr;
    ck_args.context_len_ptr   = nullptr;
    ck_args.page_table_ptr    = nullptr;
    ck_args.kscale_ptr        = nullptr;
    ck_args.vscale_ptr        = nullptr;
    ck_args.scale_s           = scale;
    ck_args.hdim              = hdim;
    ck_args.hdim_v            = hdim;
    ck_args.batch_q           = batch;
    ck_args.batch_kv          = batch;
    ck_args.batch_ratio_kv    = 1;
    ck_args.seqlen_q          = seqlen_q;
    ck_args.seqlen_kv         = seqlen_k;
    ck_args.nhead_q           = nhead_q;
    ck_args.nhead_kv          = nhead_k;
    ck_args.nhead_ratio_kv    = nhead_q / nhead_k;
    ck_args.page_size         = 0;
    ck_args.max_pages_per_seq = 0;
    ck_args.max_kv_tokens     = 0;

    ck_tile::stream_config sc;
    sc.stream_id_ = stream;

    const char* mode = (seqlen_q == 1) ? "Decode" : "Prefill";
    fprintf(stderr, "[CK Naive %s] batch=%d, nhead_q=%d, nhead_k=%d, "
            "seqlen_q=%d, seqlen_k=%d, hdim=%d, scale=%f\n",
            mode, batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim, scale);

    float result = ck_tile::naive_attention_fwd(traits, ck_args, sc);
    if (result < 0) {
        fprintf(stderr, "[CK Naive FMHA] naive_attention_fwd failed (result=%f)\n", result);
        return hipErrorUnknown;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[CK Naive FMHA] error: %s\n",
                hipGetErrorString(err));
        return err;
    }
    return hipSuccess;
}

#else // !HAS_CK_TILE

// Stub: satisfies the compiler when CK Tile is not available.
// Never called at runtime (use_ck_naive_impl() returns false when kHasCkTile is false).
// If reached, it indicates a logic error in the dispatch code.
static hipError_t launch_fmha_ck_naive(
    hipStream_t, const void*, const void*, const void*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float)
{
    fprintf(stderr, "[FMHA ERROR] launch_fmha_ck_naive called but CK Tile was not available at build time.\n"
                    "  This is a logic error: FMHA_USE_CK_NAIVE=1 should have no effect without CK_ROOT.\n"
                    "  Rebuild with -DCK_ROOT=/path/to/composable_kernel to enable CK Tile support.\n");
    return hipErrorNotSupported;
}

#endif // HAS_CK_TILE

// =============================================================================
// Check env var once (thread-safe lazy init)
// =============================================================================
static bool use_ck_naive_impl()
{
    static int cached = -1;
    if (cached < 0) {
        if (kHasCkTile) {
            const char* env = std::getenv("FMHA_USE_CK_NAIVE");
            cached = (env && env[0] == '1') ? 1 : 0;
            if (cached) {
                fprintf(stderr, "[FMHA] FMHA_USE_CK_NAIVE=1: using CK Tile naive_attention_fwd\n");
            } else {
                fprintf(stderr, "[FMHA] Using custom portable kernel (default). "
                        "Set FMHA_USE_CK_NAIVE=1 to use CK Tile.\n");
            }
        } else {
            cached = 0;
            fprintf(stderr, "[FMHA] Using custom portable kernel (CK Tile not available at build time)\n");
        }
    }
    return cached == 1;
}

// =============================================================================
// Public C API — dispatches based on FMHA_USE_CK_NAIVE env var
// =============================================================================
extern "C"
hipError_t launch_fmha_fwd_fp16_causal(
    hipStream_t stream,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    int32_t batch,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t hdim,
    float scale)
{
    if (use_ck_naive_impl()) {
        return launch_fmha_ck_naive(
            stream, q_ptr, k_ptr, v_ptr, o_ptr,
            batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim, scale);
    }

    return launch_fmha_portable(
        stream, q_ptr, k_ptr, v_ptr, o_ptr,
        batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim, scale);
}
