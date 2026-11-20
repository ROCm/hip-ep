/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- scaled_masked_softmax_kernel.hip - softmax(mask ? x*s : fill) ------===//
//
// Last-axis softmax over scores laid out [outer, R, C], with the score scale
// and an id-equality mask applied on the way in:
//
//   out[o, r, c] = softmax_c( row_ids[r] == col_ids[c] ? x[o, r, c] * scale
//                                                      : fill )
//
// This is what hip.mul(scores, scale) -> hip.where(row_ids[R,1] ==
// col_ids[1,C], ., fill) -> hip.miopen.softmax computes, without writing the
// scaled tensor, the R x C mask or the masked tensor. An id vector of length 1
// broadcasts, as it does through hip.where.
//
// The scaled value is rounded to the element type before the softmax sees it,
// because the separate hip.mul stored it in that type.
//
// One block per row position r. The mask for that position is the same for
// every o, so each thread decodes its slice of it once into a bit set and then
// walks the `outer` rows sharing it, holding its slice of each row in
// registers: one read and one write per element. fp16 rows that are 16-byte
// aligned move in 16-byte vectors; other rows use 2- or 4-byte elements. Rows
// wider than the register form fall back to a strided three-pass kernel that
// tests the ids per element.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

#include "hip_custom_kernels.h"

#ifndef WAVE_SIZE
#if defined(__GFX9__) || defined(__GFX9_NEXT__)
#define WAVE_SIZE 64
#else
#define WAVE_SIZE 32
#endif
#endif

#ifndef LOG2E
#define LOG2E 1.44269504088896340736f
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kMaxRegElems = 32; // one bit per element in a uint32_t

__device__ inline float warp_max(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v = fmaxf(v, __shfl_xor(v, off));
  return v;
}

__device__ inline float warp_sum(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v += __shfl_xor(v, off);
  return v;
}

// Each reduction broadcasts through its own LDS cell (`slot`); reusing one cell
// for both lets a fast wave overwrite a result a slow wave has not read yet.
__device__ inline float block_max(float v, float *smem, int tid, int waves,
                                  int slot) {
  v = warp_max(v);
  if (tid % WAVE_SIZE == 0)
    smem[tid / WAVE_SIZE] = v;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float t = tid < waves ? smem[tid] : -INFINITY;
    t = warp_max(t);
    if (tid == 0)
      smem[slot] = t;
  }
  __syncthreads();
  return smem[slot];
}

__device__ inline float block_sum(float v, float *smem, int tid, int waves,
                                  int slot) {
  v = warp_sum(v);
  if (tid % WAVE_SIZE == 0)
    smem[tid / WAVE_SIZE] = v;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float t = tid < waves ? smem[tid] : 0.0f;
    t = warp_sum(t);
    if (tid == 0)
      smem[slot] = t;
  }
  __syncthreads();
  return smem[slot];
}

__device__ inline float load_f(const __half *p) { return __half2float(*p); }
__device__ inline float load_f(const float *p) { return *p; }
__device__ inline void store_f(__half *p, float v) { *p = __float2half(v); }
__device__ inline void store_f(float *p, float v) { *p = v; }

// x * scale, rounded to T as the standalone multiply stored it.
__device__ inline float scaled(__half x, float scale) {
  return __half2float(__float2half(__half2float(x) * scale));
}
__device__ inline float scaled(float x, float scale) { return x * scale; }

template <typename T, int ELEMS>
__global__ void __launch_bounds__(kThreads)
    scaled_masked_softmax_regs_kernel(const T *in, T *out,
                                      const int64_t *row_ids, int row_ids_len,
                                      const int64_t *col_ids, int col_ids_len,
                                      int outer, int R, int C, float scale,
                                      float fill) {
  __shared__ float smem[kThreads / WAVE_SIZE + 2];
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int waves = kThreads / WAVE_SIZE;

  const int64_t row_id = row_ids[row_ids_len == 1 ? 0 : r];
  uint32_t keep = 0;
#pragma unroll
  for (int k = 0; k < ELEMS; ++k) {
    const int c = tid + k * kThreads;
    if (c < C && col_ids[col_ids_len == 1 ? 0 : c] == row_id)
      keep |= 1u << k;
  }

  for (int o = 0; o < outer; ++o) {
    const size_t base = ((size_t)o * R + r) * (size_t)C;
    float x[ELEMS];
    float local_max = -INFINITY;
#pragma unroll
    for (int k = 0; k < ELEMS; ++k) {
      const int c = tid + k * kThreads;
      float v = -INFINITY;
      if (c < C)
        v = (keep >> k) & 1u ? scaled(in[base + c], scale) : fill;
      x[k] = v;
      local_max = fmaxf(local_max, v);
    }
    const float row_max = block_max(local_max, smem, tid, waves, waves);

    float local_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < ELEMS; ++k) {
      const int c = tid + k * kThreads;
      x[k] = c < C ? exp2f((x[k] - row_max) * LOG2E) : 0.0f;
      local_sum += x[k];
    }
    const float inv = 1.0f / block_sum(local_sum, smem, tid, waves, waves + 1);

#pragma unroll
    for (int k = 0; k < ELEMS; ++k) {
      const int c = tid + k * kThreads;
      if (c < C)
        store_f(&out[base + c], x[k] * inv);
    }
  }
}

// fp16 rows in 16-byte vectors (8 halves per lane, lane vectors strided by
// the block so each load stays coalesced). A vector whose 8 columns are all
// masked is not read at all. One 7296-wide image (every id equal): 16.3 vs
// 18.0 ms per 16x7296x7296 call against the per-element form above; two
// frames: 13.0 vs 13.6 ms.
template <int NV>
__global__ void __launch_bounds__(kThreads)
    scaled_masked_softmax_vec_kernel(const __half *in, __half *out,
                                     const int64_t *row_ids, int row_ids_len,
                                     const int64_t *col_ids, int col_ids_len,
                                     int outer, int R, int C, float scale,
                                     float fill) {
  __shared__ float smem[kThreads / WAVE_SIZE + 2];
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int waves = kThreads / WAVE_SIZE;
  const int nvec = C / 8;

  const int64_t row_id = row_ids[row_ids_len == 1 ? 0 : r];
  uint32_t keep[NV];
#pragma unroll
  for (int k = 0; k < NV; k++) {
    keep[k] = 0;
    const int v = tid + k * kThreads;
    if (v < nvec)
#pragma unroll
      for (int e = 0; e < 8; e++)
        if (col_ids[col_ids_len == 1 ? 0 : v * 8 + e] == row_id)
          keep[k] |= 1u << e;
  }

  for (int o = 0; o < outer; o++) {
    const size_t base = ((size_t)o * R + r) * (size_t)C;
    const uint4 *src = reinterpret_cast<const uint4 *>(in + base);
    uint4 *dst = reinterpret_cast<uint4 *>(out + base);
    float x[NV][8];
    float local_max = -INFINITY;
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int v = tid + k * kThreads;
      if (v < nvec && keep[k]) {
        const uint4 u = src[v];
        const __half *h = reinterpret_cast<const __half *>(&u);
#pragma unroll
        for (int e = 0; e < 8; e++)
          x[k][e] = (keep[k] >> e) & 1u ? scaled(h[e], scale) : fill;
      } else {
#pragma unroll
        for (int e = 0; e < 8; e++)
          x[k][e] = v < nvec ? fill : -INFINITY;
      }
#pragma unroll
      for (int e = 0; e < 8; e++)
        local_max = fmaxf(local_max, x[k][e]);
    }
    const float row_max = block_max(local_max, smem, tid, waves, waves);

    float local_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; k++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        x[k][e] = exp2f((x[k][e] - row_max) * LOG2E);
        local_sum += x[k][e];
      }
    const float inv = 1.0f / block_sum(local_sum, smem, tid, waves, waves + 1);

#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int v = tid + k * kThreads;
      if (v < nvec) {
        uint4 u;
        __half *h = reinterpret_cast<__half *>(&u);
#pragma unroll
        for (int e = 0; e < 8; e++)
          h[e] = __float2half(x[k][e] * inv);
        dst[v] = u;
      }
    }
  }
}

// Rows too wide for the register form: one block per row, three strided
// passes. Pass 2 parks exp(v - max) in `out`, which is the buffer pass 3
// normalizes; a thread only ever re-reads the elements it wrote itself, so
// `out` may alias `in`.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    scaled_masked_softmax_strided_kernel(const T *in, T *out,
                                         const int64_t *row_ids,
                                         int row_ids_len,
                                         const int64_t *col_ids,
                                         int col_ids_len, int R, int C,
                                         float scale, float fill) {
  __shared__ float smem[kThreads / WAVE_SIZE + 2];
  const size_t row = blockIdx.x;
  const int r = static_cast<int>(row % R);
  const int tid = threadIdx.x;
  const int waves = kThreads / WAVE_SIZE;
  const int64_t row_id = row_ids[row_ids_len == 1 ? 0 : r];
  const T *in_row = in + row * (size_t)C;
  T *out_row = out + row * (size_t)C;

  auto value = [&](int c) {
    return col_ids[col_ids_len == 1 ? 0 : c] == row_id
               ? scaled(in_row[c], scale)
               : fill;
  };

  float local_max = -INFINITY;
  for (int c = tid; c < C; c += kThreads)
    local_max = fmaxf(local_max, value(c));
  const float row_max = block_max(local_max, smem, tid, waves, waves);

  float local_sum = 0.0f;
  for (int c = tid; c < C; c += kThreads) {
    const float e = exp2f((value(c) - row_max) * LOG2E);
    store_f(&out_row[c], e);
    local_sum += e;
  }
  const float inv = 1.0f / block_sum(local_sum, smem, tid, waves, waves + 1);

  for (int c = tid; c < C; c += kThreads)
    store_f(&out_row[c], load_f(&out_row[c]) * inv);
}

template <typename T>
int launch(hipStream_t stream, const T *in, T *out, const int64_t *row_ids,
           int row_ids_len, const int64_t *col_ids, int col_ids_len, int outer,
           int R, int C, float scale, float fill) {
  if constexpr (std::is_same_v<T, __half>) {
    const int nvec = C / 8;
    const bool vec_ok =
        C % 8 == 0 && nvec <= 4 * kThreads &&
        reinterpret_cast<uintptr_t>(in) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(out) % 16 == 0;
    if (vec_ok) {
      const int nv = (nvec + kThreads - 1) / kThreads;
#define HIP_SMS_VEC(NV)                                                        \
  scaled_masked_softmax_vec_kernel<NV><<<R, kThreads, 0, stream>>>(            \
      in, out, row_ids, row_ids_len, col_ids, col_ids_len, outer, R, C, scale, \
      fill)
      if (nv <= 1)
        HIP_SMS_VEC(1);
      else if (nv <= 2)
        HIP_SMS_VEC(2);
      else
        HIP_SMS_VEC(4);
#undef HIP_SMS_VEC
      return hipGetLastError() == hipSuccess ? 0 : -1;
    }
  }
  const int elems = (C + kThreads - 1) / kThreads;
#define HIP_SMS_REGS(E)                                                        \
  scaled_masked_softmax_regs_kernel<T, E><<<R, kThreads, 0, stream>>>(         \
      in, out, row_ids, row_ids_len, col_ids, col_ids_len, outer, R, C, scale, \
      fill)
  if (elems <= 1)
    HIP_SMS_REGS(1);
  else if (elems <= 2)
    HIP_SMS_REGS(2);
  else if (elems <= 4)
    HIP_SMS_REGS(4);
  else if (elems <= 8)
    HIP_SMS_REGS(8);
  else if (elems <= 16)
    HIP_SMS_REGS(16);
  else if (elems <= kMaxRegElems)
    HIP_SMS_REGS(32);
  else
    scaled_masked_softmax_strided_kernel<T>
        <<<(unsigned)((size_t)outer * R), kThreads, 0, stream>>>(
            in, out, row_ids, row_ids_len, col_ids, col_ids_len, R, C, scale,
            fill);
#undef HIP_SMS_REGS
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // namespace

extern "C" int hip_softmax_row_scaled_masked(
    void *stream_ptr, const void *input, void *output, const int64_t *row_ids,
    int64_t row_ids_len, const int64_t *col_ids, int64_t col_ids_len,
    int64_t outer, int64_t R, int64_t C, float scale, float fill,
    int elem_size_bytes) {
  if (!input || !output || !row_ids || !col_ids || outer <= 0 || R <= 0 ||
      C <= 0)
    return -1;
  // hip.where broadcasts a mask axis of extent 1; any other mismatch is a
  // malformed graph, not something to clamp.
  if ((row_ids_len != 1 && row_ids_len != R) ||
      (col_ids_len != 1 && col_ids_len != C))
    return -1;
  if (outer * R > INT32_MAX || C > INT32_MAX)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();
  if (elem_size_bytes == 2)
    return launch(stream, static_cast<const __half *>(input),
                  static_cast<__half *>(output), row_ids, (int)row_ids_len,
                  col_ids, (int)col_ids_len, (int)outer, (int)R, (int)C, scale,
                  fill);
  if (elem_size_bytes == 4)
    return launch(stream, static_cast<const float *>(input),
                  static_cast<float *>(output), row_ids, (int)row_ids_len,
                  col_ids, (int)col_ids_len, (int)outer, (int)R, (int)C, scale,
                  fill);
  return -1;
}
