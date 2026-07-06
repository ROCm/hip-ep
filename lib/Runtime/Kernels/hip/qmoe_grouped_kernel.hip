/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Batched QMoE prefill plumbing (num_tokens > 1).
//
// The serial per-expert loop runs gather->fc1->swiglu->fc2->scatter as a
// dependency chain, per expert, serialized across ~all 256 experts. Measured
// (launch-blocking A/B) that this chain is overlap-limited: the independent
// per-expert fc1 (and fc2) matmuls cannot overlap because tiny plumbing kernels
// sit between them. Batching the plumbing onto the bucket-sorted layout lets the
// restructured flow be: one gather -> ALL fc1 matmuls (back-to-back, overlap) ->
// one SwiGLU -> ALL fc2 matmuls -> one weighted atomic scatter -> one cast.
//
// This file holds the two plumbing kernels that differ from the serial path:
//   * hip_qmoe_scatter_add_atomic : fc2_out[total_rows,hidden] -> fp32 output
//         accumulator, weighted by the per-row routing weight. A token routes to
//         k experts (k rows) -> concurrent writes -> atomicAdd(float), fp32 accum.
//   * hip_qmoe_cast_f32_f16       : fp32 accumulator -> fp16 output.
// (Gather reuses hip_qmoe_gather_tokens; SwiGLU reuses hip_qmoe_swiglu.)
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// ===========================================================================
// Streaming grouped int4 GEMV (the "mega-kernel").
//
// One launch computes, for every expert e, C_e = A_e x dequant(W_e) over that
// expert's contiguous bucket-sorted rows. Design goal: read each expert's int4
// weights ONCE at high bandwidth (like the decode-fused kernel that hits
// ~decode-class GB/s), reusing them across BM rows via register accumulators.
// NO WMMA (its smem staging + 16-row min M-waste hurt at M~6); NO per-row
// re-read (weight reuse across BM rows); ONE launch (device tile schedule).
//
// Grid: (ceil(N/TILE_N), max_m_tiles). blockIdx.y -> (expert, local m-tile) via
// the schedule; padding tiles early-exit on *total_m_tiles. Each block: stream
// the TILE_N weight columns over K in uint4 (32-nibble) chunks ONCE; for each of
// the up-to-BM rows, factored-dequant dot ((raw - a_sum*zp)*scale). Threads split
// K and warp/LDS-reduce. Writes are disjoint across blocks (distinct rows/cols)
// -> no atomics. Packed-nibble zp read in-kernel (raw QMoE layout) or default 8.
// ===========================================================================
template <int BS, int BM, int TILE_N>
__global__ void qmoe_stream_gemm_kernel(
    const __half *__restrict__ A,          // [total_rows, K]
    const uint8_t *__restrict__ Wbase,     // [E, N, K/2]
    const __half *__restrict__ Sbase,      // [E, N, ngk]
    const uint8_t *__restrict__ Zbase,     // [E, N, ceil(ngk/2)] or null
    const __half *__restrict__ Bbase,      // [E, N] or null
    __half *__restrict__ C,                // [total_rows, N]
    const int *__restrict__ tile_expert,
    const int *__restrict__ tile_localm,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ expert_counts,
    const int *__restrict__ total_m_tiles,
    int N, int K, int ngk, int block_size) {
  const int by = blockIdx.y;
  if (by >= *total_m_tiles)
    return;
  const int e = tile_expert[by];
  const int localm = tile_localm[by];
  const int row0 = expert_offsets[e] + localm * BM;
  const int M = expert_counts[e];
  int rows_valid = M - localm * BM;
  if (rows_valid > BM) rows_valid = BM;
  if (rows_valid <= 0)
    return;
  const int n_base = blockIdx.x * TILE_N;
  const int tid = threadIdx.x;

  const size_t W_row_bytes = static_cast<size_t>(K) / 2;
  const uint8_t *W = Wbase + static_cast<size_t>(e) * N * W_row_bytes;
  const __half *S = Sbase + static_cast<size_t>(e) * N * ngk;
  const int pzs = (ngk + 1) / 2;
  const uint8_t *Z = Zbase ? Zbase + static_cast<size_t>(e) * N * pzs : nullptr;
  const __half *Bv = Bbase ? Bbase + static_cast<size_t>(e) * N : nullptr;
  const bool has_zp = (Z != nullptr);
  const int bs_shift = __builtin_ctz(static_cast<unsigned>(block_size));

  // Per-column weight/scale/zp row pointers for the TILE_N columns.
  const uint8_t *W_rows[TILE_N];
  const __half *S_rows[TILE_N];
  const uint8_t *Z_rows[TILE_N];
#pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    int n = n_base + t;
    int sn = (n < N) ? n : 0;
    W_rows[t] = W + static_cast<size_t>(sn) * W_row_bytes;
    S_rows[t] = S + static_cast<size_t>(sn) * ngk;
    Z_rows[t] = Z ? Z + static_cast<size_t>(sn) * pzs : nullptr;
  }

  float partial[BM][TILE_N];
#pragma unroll
  for (int r = 0; r < BM; r++)
#pragma unroll
    for (int t = 0; t < TILE_N; t++)
      partial[r][t] = 0.0f;

  const int num_u128 = K / 32;
  for (int idx = tid; idx < num_u128; idx += BS) {
    const int k32 = idx * 32;
    const int grp = k32 >> bs_shift;
    uint4 b_pk[TILE_N];
    float sv[TILE_N];
    float zp[TILE_N];
#pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      b_pk[t] = *reinterpret_cast<const uint4 *>(&W_rows[t][idx * 16]);
      sv[t] = static_cast<float>(S_rows[t][grp]);
      if (has_zp) {
        uint8_t zb = Z_rows[t][grp >> 1];
        zp[t] = static_cast<float>((grp & 1) ? (zb >> 4) : (zb & 0x0F));
      } else {
        zp[t] = 8.0f;
      }
    }
#pragma unroll
    for (int r = 0; r < BM; r++) {
      if (r >= rows_valid) break;
      const __half *a_row = A + static_cast<size_t>(row0 + r) * K + k32;
      float a_vals[32];
      float a_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; i++) {
        a_vals[i] = static_cast<float>(a_row[i]);
        a_sum += a_vals[i];
      }
#pragma unroll
      for (int t = 0; t < TILE_N; t++) {
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++)
          dot = __fmaf_rn(a_vals[i],
                          static_cast<float>((b_pk[t].x >> (i * 4)) & 0xF), dot);
#pragma unroll
        for (int i = 0; i < 8; i++)
          dot = __fmaf_rn(a_vals[8 + i],
                          static_cast<float>((b_pk[t].y >> (i * 4)) & 0xF), dot);
#pragma unroll
        for (int i = 0; i < 8; i++)
          dot = __fmaf_rn(a_vals[16 + i],
                          static_cast<float>((b_pk[t].z >> (i * 4)) & 0xF), dot);
#pragma unroll
        for (int i = 0; i < 8; i++)
          dot = __fmaf_rn(a_vals[24 + i],
                          static_cast<float>((b_pk[t].w >> (i * 4)) & 0xF), dot);
        partial[r][t] += (dot - a_sum * zp[t]) * sv[t];
      }
    }
  }

  // Reduce partial[r][t] across the BS threads (K-split): warp shuffle then LDS.
  constexpr int WARP = 32;
  const int warp_id = tid / WARP;
  const int lane = tid % WARP;
  constexpr int NW = BS / WARP;
#pragma unroll
  for (int r = 0; r < BM; r++)
#pragma unroll
    for (int t = 0; t < TILE_N; t++)
      for (int off = WARP >> 1; off > 0; off >>= 1)
        partial[r][t] += __shfl_down(partial[r][t], off);

  __shared__ float red[BM][TILE_N][NW];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < BM; r++)
#pragma unroll
      for (int t = 0; t < TILE_N; t++)
        red[r][t][warp_id] = partial[r][t];
  }
  __syncthreads();
  // First NW threads finish the cross-warp reduction and write out.
  if (warp_id == 0) {
#pragma unroll
    for (int r = 0; r < BM; r++) {
      if (r >= rows_valid) break;
#pragma unroll
      for (int t = 0; t < TILE_N; t++) {
        int n = n_base + t;
        if (n >= N) continue;
        float v = (lane < NW) ? red[r][t][lane] : 0.0f;
        for (int off = WARP >> 1; off > 0; off >>= 1)
          v += __shfl_down(v, off);
        if (lane == 0) {
          if (Bv) v += static_cast<float>(Bv[n]);
          C[static_cast<size_t>(row0 + r) * N + n] = static_cast<__half>(v);
        }
      }
    }
  }
}

// Device-side tile schedule: per-expert ceil(count_e/BM) (expert, local-m-tile)
// pairs into tile_expert/tile_localm + total count. Single-thread (num_experts
// small) -> no host readback / sync on the critical path.
__global__ void qmoe_build_schedule_kernel(
    const int32_t *__restrict__ counts, int num_experts, int BM,
    int32_t *__restrict__ tile_expert, int32_t *__restrict__ tile_localm,
    int32_t *__restrict__ total_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  int T = 0;
  for (int e = 0; e < num_experts; e++) {
    int c = counts[e];
    int mt = (c + BM - 1) / BM;
    for (int i = 0; i < mt; i++) {
      tile_expert[T] = e;
      tile_localm[T] = i;
      T++;
    }
  }
  *total_out = T;
}

extern "C" int hip_qmoe_build_schedule(void *stream, const void *counts,
                                       int64_t num_experts, int64_t bm,
                                       void *tile_expert, void *tile_localm,
                                       void *total_out) {
  if (num_experts <= 0)
    return 0;
  hipLaunchKernelGGL(qmoe_build_schedule_kernel, dim3(1), dim3(1), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const int32_t *>(counts),
                     static_cast<int>(num_experts), static_cast<int>(bm),
                     static_cast<int32_t *>(tile_expert),
                     static_cast<int32_t *>(tile_localm),
                     static_cast<int32_t *>(total_out));
  hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_build_schedule launch: %s\n",
            hipGetErrorString(e));
  return static_cast<int>(e);
}

// Streaming grouped GEMV launcher. BM=8, TILE_N=4, BS=256 (K%32==0 required;
// hidden=2048/inter=512 satisfy it). The host schedule MUST use the same BM.
extern "C" int hip_qmoe_stream_gemm(
    void *stream, const void *A, const void *Wbase, const void *Sbase,
    const void *Zbase, const void *Bbase, void *C, int64_t N, int64_t K,
    int64_t group_size, int64_t max_m_tiles, const void *tile_expert,
    const void *tile_localm, const void *expert_offsets,
    const void *expert_counts, const void *total_m_tiles) {
  if (max_m_tiles <= 0)
    return 0;
  if ((K % 32) != 0 || group_size <= 0) {
    fprintf(stderr, "[custom_kernels] hip_qmoe_stream_gemm: bad shape (K=%lld "
                    "gs=%lld)\n",
            (long long)K, (long long)group_size);
    return -1;
  }
  constexpr int BS = 256, BM = 4, TILE_N = 4;
  const int ngk = static_cast<int>((K + group_size - 1) / group_size);
  dim3 grid(static_cast<unsigned>((N + TILE_N - 1) / TILE_N),
            static_cast<unsigned>(max_m_tiles));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qmoe_stream_gemm_kernel<BS, BM, TILE_N>), grid, dim3(BS),
      0, static_cast<hipStream_t>(stream), static_cast<const __half *>(A),
      static_cast<const uint8_t *>(Wbase), static_cast<const __half *>(Sbase),
      static_cast<const uint8_t *>(Zbase), static_cast<const __half *>(Bbase),
      static_cast<__half *>(C), static_cast<const int *>(tile_expert),
      static_cast<const int *>(tile_localm),
      static_cast<const int *>(expert_offsets),
      static_cast<const int *>(expert_counts),
      static_cast<const int *>(total_m_tiles), static_cast<int>(N),
      static_cast<int>(K), ngk, static_cast<int>(group_size));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_stream_gemm launch: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

__global__ void qmoe_scatter_add_atomic_kernel(
    float *__restrict__ output_accum,        // [num_tokens, width] fp32
    const __half *__restrict__ expert_out,   // [count, width]
    const int32_t *__restrict__ token_ids,   // [count]
    const __half *__restrict__ weights,      // [count]
    int64_t width, int64_t count) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count * width)
    return;
  int64_t row = idx / width;
  int64_t col = idx % width;
  int dst = token_ids[row];
  float w = static_cast<float>(weights[row]);
  float v = static_cast<float>(expert_out[idx]);
  atomicAdd(&output_accum[static_cast<size_t>(dst) * width + col], w * v);
}

__global__ void qmoe_cast_f32_f16_kernel(const float *__restrict__ in,
                                         __half *__restrict__ out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = static_cast<__half>(in[i]);
}

extern "C" int hip_qmoe_scatter_add_atomic(void *stream, void *output_accum,
                                           const void *expert_out,
                                           const void *token_ids,
                                           const void *weights, int64_t width,
                                           int64_t count) {
  if (count <= 0 || width <= 0)
    return 0;
  constexpr int BS = 256;
  int64_t total = count * width;
  dim3 grid(static_cast<unsigned>((total + BS - 1) / BS));
  hipLaunchKernelGGL(qmoe_scatter_add_atomic_kernel, grid, dim3(BS), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<float *>(output_accum),
                     static_cast<const __half *>(expert_out),
                     static_cast<const int32_t *>(token_ids),
                     static_cast<const __half *>(weights), width, count);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_scatter_add_atomic launch: %s\n",
            hipGetErrorString(e));
  return static_cast<int>(e);
}

extern "C" int hip_qmoe_cast_f32_f16(void *stream, const void *in, void *out,
                                     int64_t n) {
  if (n <= 0)
    return 0;
  constexpr int BS = 256;
  dim3 grid(static_cast<unsigned>((n + BS - 1) / BS));
  hipLaunchKernelGGL(qmoe_cast_f32_f16_kernel, grid, dim3(BS), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const float *>(in), static_cast<__half *>(out),
                     n);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_cast_f32_f16 launch: %s\n",
            hipGetErrorString(e));
  return static_cast<int>(e);
}
