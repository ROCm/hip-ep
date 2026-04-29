/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * High-performance RoPE (Rotary Positional Embedding) kernel for FP16.
 * Optimized for RDNA 3.5/4 (gfx1150/gfx1151, wave32).
 *
 * Key optimizations vs rope_kernel.hip:
 *   1. One block per (batch, seq) position — ALL heads in one block
 *      → 32x fewer blocks, eliminates scheduling overhead at large S
 *   2. cos/sin cached in LDS (shared memory)
 *      → all heads share the same cos/sin, loaded once from global
 *   3. __half2 vectorized loads/stores for rotary computation
 *   4. Bit-shift indexing (no integer division in hot loop)
 *   5. Block size 256 → 8 waves, optimal RDNA occupancy
 *   6. Single kernel handles rotary + passthrough (no copy kernel)
 *   7. __launch_bounds__ for register allocation hints
 *
 * API: hip_rope_forward_opt() — same signature as hip_rope_forward()
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

/* ======================================================================
 * Half-rotated RoPE (GPT-NeoX / LLaMA style)
 *
 * Grid:  (B * S)       — one block per sequence position
 * Block: (256)          — 8 waves, optimal for RDNA wave32
 * LDS:   cos[half_rot] + sin[half_rot]
 *
 * All heads for a given position share cos/sin → cached in LDS.
 * Threads iterate over (head, half2_pair) tuples via bit-shift indexing.
 * ====================================================================== */
__launch_bounds__(256)
__global__ void rope_half_rotated_opt(
    const __half* __restrict__ input,
    const int64_t* __restrict__ position_ids,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    __half* __restrict__ output,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int half_rot,
    const int half2_shift,
    const int max_seq_len)
{
    const int bs_idx = blockIdx.x;
    const int b = bs_idx / seq_len;
    const int s = bs_idx % seq_len;

    int64_t pos_raw = position_ids[b * seq_len + s];
    int pos = static_cast<int>(pos_raw < 0 ? 0 :
                  (pos_raw >= max_seq_len ? max_seq_len - 1 : pos_raw));

    // ---- Load cos/sin into LDS (shared by all heads) ----
    extern __shared__ __half smem[];
    __half* s_cos = smem;
    __half* s_sin = smem + half_rot;

    {
        const __half* g_cos = cos_cache + pos * half_rot;
        const __half* g_sin = sin_cache + pos * half_rot;
        const int h2_load = half_rot >> 1;
        for (int i = threadIdx.x; i < h2_load; i += blockDim.x) {
            reinterpret_cast<__half2*>(s_cos)[i] =
                reinterpret_cast<const __half2*>(g_cos)[i];
            reinterpret_cast<__half2*>(s_sin)[i] =
                reinterpret_cast<const __half2*>(g_sin)[i];
        }
        if ((half_rot & 1) && threadIdx.x == 0) {
            s_cos[half_rot - 1] = g_cos[half_rot - 1];
            s_sin[half_rot - 1] = g_sin[half_rot - 1];
        }
    }
    __syncthreads();

    const int base = bs_idx * num_heads * head_dim;
    const int half2_count = half_rot >> 1;
    const int half2_mask  = half2_count - 1;
    const int total_h2    = num_heads * half2_count;

    // ---- Rotary: all heads, __half2 vectorized ----
    for (int idx = threadIdx.x; idx < total_h2; idx += blockDim.x) {
        const int h = idx >> half2_shift;
        const int d = (idx & half2_mask) << 1;
        const int off = base + h * head_dim;

        __half2 x0 = *reinterpret_cast<const __half2*>(&input[off + d]);
        __half2 x1 = *reinterpret_cast<const __half2*>(&input[off + d + half_rot]);
        __half2 c2 = *reinterpret_cast<const __half2*>(&s_cos[d]);
        __half2 s2 = *reinterpret_cast<const __half2*>(&s_sin[d]);

        *reinterpret_cast<__half2*>(&output[off + d]) =
            __hsub2(__hmul2(x0, c2), __hmul2(x1, s2));
        *reinterpret_cast<__half2*>(&output[off + d + half_rot]) =
            __hadd2(__hmul2(x0, s2), __hmul2(x1, c2));
    }

    // Scalar tail for odd half_rot
    if ((half_rot & 1) && threadIdx.x < num_heads) {
        const int d = half_rot - 1;
        const int off = base + threadIdx.x * head_dim;
        __half x0 = input[off + d], x1 = input[off + d + half_rot];
        __half cv = s_cos[d], sv = s_sin[d];
        output[off + d]            = __hsub(__hmul(x0, cv), __hmul(x1, sv));
        output[off + d + half_rot] = __hadd(__hmul(x0, sv), __hmul(x1, cv));
    }

    // ---- Passthrough: copy dimensions beyond rotary_dim ----
    const int rot_full = half_rot << 1;
    const int pass = head_dim - rot_full;
    if (pass > 0) {
        const int pass_h2 = pass >> 1;
        const int total_pass = num_heads * pass_h2;
        for (int idx = threadIdx.x; idx < total_pass; idx += blockDim.x) {
            int h = idx / pass_h2;
            int t = idx - h * pass_h2;
            int d = rot_full + (t << 1);
            int off = base + h * head_dim;
            *reinterpret_cast<__half2*>(&output[off + d]) =
                *reinterpret_cast<const __half2*>(&input[off + d]);
        }
        if ((pass & 1) && threadIdx.x < num_heads) {
            int off = base + threadIdx.x * head_dim;
            output[off + head_dim - 1] = input[off + head_dim - 1];
        }
    }
}

/* ======================================================================
 * Interleaved RoPE (GPT-J style)
 *
 * Pair: (2d, 2d+1) for d in [0, half_rot)
 * Same grid/block strategy: one block per position, all heads.
 * ====================================================================== */
__launch_bounds__(256)
__global__ void rope_interleaved_opt(
    const __half* __restrict__ input,
    const int64_t* __restrict__ position_ids,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    __half* __restrict__ output,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int half_rot,
    const int hr_shift,
    const int max_seq_len)
{
    const int bs_idx = blockIdx.x;
    const int b = bs_idx / seq_len;
    const int s = bs_idx % seq_len;

    int64_t pos_raw = position_ids[b * seq_len + s];
    int pos = static_cast<int>(pos_raw < 0 ? 0 :
                  (pos_raw >= max_seq_len ? max_seq_len - 1 : pos_raw));

    extern __shared__ __half smem[];
    __half* s_cos = smem;
    __half* s_sin = smem + half_rot;

    {
        const __half* g_cos = cos_cache + pos * half_rot;
        const __half* g_sin = sin_cache + pos * half_rot;
        const int h2_load = half_rot >> 1;
        for (int i = threadIdx.x; i < h2_load; i += blockDim.x) {
            reinterpret_cast<__half2*>(s_cos)[i] =
                reinterpret_cast<const __half2*>(g_cos)[i];
            reinterpret_cast<__half2*>(s_sin)[i] =
                reinterpret_cast<const __half2*>(g_sin)[i];
        }
        if ((half_rot & 1) && threadIdx.x == 0) {
            s_cos[half_rot - 1] = g_cos[half_rot - 1];
            s_sin[half_rot - 1] = g_sin[half_rot - 1];
        }
    }
    __syncthreads();

    const int base = bs_idx * num_heads * head_dim;
    const int hr_mask = half_rot - 1;
    const int total_work = num_heads * half_rot;

    // ---- Rotary: all heads ----
    for (int idx = threadIdx.x; idx < total_work; idx += blockDim.x) {
        const int h = idx >> hr_shift;
        const int d = idx & hr_mask;
        const int off = base + h * head_dim;

        __half2 x_pair = *reinterpret_cast<const __half2*>(&input[off + d * 2]);
        __half x0 = __low2half(x_pair);
        __half x1 = __high2half(x_pair);

        __half cv = s_cos[d];
        __half sv = s_sin[d];

        __half out0 = __hsub(__hmul(x0, cv), __hmul(x1, sv));
        __half out1 = __hadd(__hmul(x0, sv), __hmul(x1, cv));

        *reinterpret_cast<__half2*>(&output[off + d * 2]) =
            __halves2half2(out0, out1);
    }

    // ---- Passthrough ----
    const int rot_full = half_rot << 1;
    const int pass = head_dim - rot_full;
    if (pass > 0) {
        const int pass_h2 = pass >> 1;
        const int total_pass = num_heads * pass_h2;
        for (int idx = threadIdx.x; idx < total_pass; idx += blockDim.x) {
            int h = idx / pass_h2;
            int t = idx - h * pass_h2;
            int d = rot_full + (t << 1);
            int off = base + h * head_dim;
            *reinterpret_cast<__half2*>(&output[off + d]) =
                *reinterpret_cast<const __half2*>(&input[off + d]);
        }
        if ((pass & 1) && threadIdx.x < num_heads) {
            int off = base + threadIdx.x * head_dim;
            output[off + head_dim - 1] = input[off + head_dim - 1];
        }
    }
}

extern "C" int hip_rope_forward_opt(
    void* stream_ptr,
    const void* input,
    const void* position_ids,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_seq_len_i64,
    int64_t interleaved,
    int64_t element_size_bytes)
{
    if (!input || !position_ids || !cos_cache || !sin_cache || !output) {
        fprintf(stderr, "hip_rope_forward_opt: null pointer argument\n");
        return -1;
    }

    if (element_size_bytes != 2) {
        fprintf(stderr,
                "hip_rope_forward_opt: only fp16 supported, got elem_size=%lld\n",
                (long long)element_size_bytes);
        return -1;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

    const int bs  = static_cast<int>(batch_size);
    const int sl  = static_cast<int>(seq_len);
    const int nh  = static_cast<int>(num_heads);
    const int hd  = static_cast<int>(head_dim);
    const int rd  = static_cast<int>(rotary_dim);
    const int hr  = rd / 2;
    const int msl = static_cast<int>(max_seq_len_i64);

    const auto* in   = static_cast<const __half*>(input);
    const auto* pos  = static_cast<const int64_t*>(position_ids);
    const auto* cosc = static_cast<const __half*>(cos_cache);
    const auto* sinc = static_cast<const __half*>(sin_cache);
    auto*       out  = static_cast<__half*>(output);

    const int smem_bytes = 2 * hr * static_cast<int>(sizeof(__half));

    dim3 grid(bs * sl);
    dim3 block(256);

    if (interleaved) {
        int hr_shift = 0;
        { int v = hr; while (v > 1) { v >>= 1; hr_shift++; } }

        hipLaunchKernelGGL(rope_interleaved_opt,
            grid, block, smem_bytes, stream,
            in, pos, cosc, sinc, out,
            sl, nh, hd, hr, hr_shift, msl);
    } else {
        int half2_count = hr >> 1;
        int half2_shift = 0;
        { int v = half2_count; while (v > 1) { v >>= 1; half2_shift++; } }

        hipLaunchKernelGGL(rope_half_rotated_opt,
            grid, block, smem_bytes, stream,
            in, pos, cosc, sinc, out,
            sl, nh, hd, hr, half2_shift, msl);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hip_rope_forward_opt: launch failed: %s\n",
                hipGetErrorString(err));
        return -1;
    }
    return 0;
}
