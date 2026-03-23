/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "hip_custom_kernels.h"

static constexpr int GQA_THREADS = 256;

// =============================================================================
// KV Cache Update Kernel
// =============================================================================
// Writes new K/V from BSHD [B,sq,G,d] into BNSD KV cache [B,G,max_seq,d]
// at positions [past_len .. past_len+sq). Fuses the layout transpose.

__global__ void kv_cache_update_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ cache,
    int sq, int G, int d, int max_seq, int past_len) {
  int batch_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * max_seq + (past_len + s)) * d + di;
  cache[dst_idx] = src[src_idx];
}

// =============================================================================
// RoPE Kernel
// =============================================================================
// Half-rotated RoPE: for each position s, pair_d in [0, half_rot):
//   out[d] = in[d]*cos - in[d+half]*sin
//   out[d+half] = in[d+half]*cos + in[d]*sin

__global__ void rope_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    int seq_len, int num_heads, int head_dim,
    int half_rot, int past_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;
  int elems = seq_len * num_heads * half_rot;
  if (idx >= elems) return;

  int d = idx % half_rot;
  int remaining = idx / half_rot;
  int h = remaining % num_heads;
  int s = remaining / num_heads;
  if (s >= seq_len) return;

  int pos = past_len + s;
  int base = ((batch_idx * seq_len + s) * num_heads + h) * head_dim;

  float x1 = __half2float(input[base + d]);
  float x2 = __half2float(input[base + d + half_rot]);
  float c  = __half2float(cos_cache[pos * half_rot + d]);
  float sn = __half2float(sin_cache[pos * half_rot + d]);

  output[base + d]            = __float2half(x1 * c - x2 * sn);
  output[base + d + half_rot] = __float2half(x2 * c + x1 * sn);
}

// =============================================================================
// Transpose middle two dims of 4D tensor
// =============================================================================
// [B, dim1, dim2, D] -> [B, dim2, dim1, D]

__global__ void transpose_mid_dims_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int dim1, int dim2, int D) {
  int batch_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int per_batch = dim1 * dim2 * D;
  if (idx >= per_batch) return;

  int di = idx % D;
  int remaining = idx / D;
  int d2 = remaining % dim2;
  int d1 = remaining / dim2;

  int src_off = batch_idx * per_batch + idx;
  int dst_off = batch_idx * per_batch + (d2 * dim1 + d1) * D + di;
  dst[dst_off] = src[src_off];
}

// =============================================================================
// KV Group Expansion: G groups -> H heads
// =============================================================================
// For head h, copies from source group g = h / heads_per_group.

__global__ void expand_kv_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int heads_per_group,
    int src_stride,
    int dst_stride,
    int copy_elems) {
  int h = blockIdx.x;
  int g = h / heads_per_group;
  int idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (idx < copy_elems) {
    dst[(size_t)h * dst_stride + idx] =
        src[(size_t)g * src_stride + idx];
  }
}

// =============================================================================
// Causal Mask -- prefill only, skipped when sq == 1
// =============================================================================
// S[k, q] = -65504 (fp16 -inf) where k > past_len + q

__global__ void causal_mask_kernel(
    __half* __restrict__ S,
    int skv, int sq, int batch_stride, int past_len) {
  int head = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = skv * sq;
  if (idx >= total) return;

  int k = idx % skv;
  int q = idx / skv;
  if (k > past_len + q) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] =
        __float2half(-65504.0f);
  }
}

// =============================================================================
// Column-wise Softmax in-place
// =============================================================================
// One threadblock per (head, query) pair.

__global__ void softmax_inplace_kernel(
    __half* data, int rows, int cols, int batch_stride) {
  int linear = blockIdx.x;
  int head = linear / cols;
  int col = linear % cols;
  __half* col_ptr = data + (size_t)head * batch_stride + (size_t)col * rows;

  extern __shared__ float smem[];

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, __half2float(col_ptr[i]));
  smem[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < (unsigned)s)
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float max_val = smem[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < rows; i += blockDim.x) {
    float e = expf(__half2float(col_ptr[i]) - max_val);
    col_ptr[i] = __float2half(e);
    local_sum += e;
  }
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < (unsigned)s)
      smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float sum_val = smem[0];
  __syncthreads();

  float inv_sum = 1.0f / sum_val;
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    col_ptr[i] = __float2half(__half2float(col_ptr[i]) * inv_sum);
}

// =============================================================================
// Extern "C" Launchers (thin wrappers around kernel launches)
// =============================================================================

extern "C" int hip_gqa_kv_cache_update(
    void* stream_ptr, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int max_seq, int past_len) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = sq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_update_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(cache),
      sq, G, d, max_seq, past_len);
  return hipSuccess;
}

extern "C" int hip_gqa_rope(
    void* stream_ptr, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int elems = seq_len * num_heads * half_rot;
  int blocks = (elems + GQA_THREADS - 1) / GQA_THREADS;
  rope_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(input),
      static_cast<__half*>(output),
      static_cast<const __half*>(cos_cache),
      static_cast<const __half*>(sin_cache),
      seq_len, num_heads, head_dim, half_rot, past_len);
  return hipSuccess;
}

extern "C" int hip_gqa_transpose_mid_dims(
    void* stream_ptr, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int per_batch = dim1 * dim2 * D;
  int blocks = (per_batch + GQA_THREADS - 1) / GQA_THREADS;
  transpose_mid_dims_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst),
      dim1, dim2, D);
  return hipSuccess;
}

extern "C" int hip_gqa_expand_kv(
    void* stream_ptr, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int blocksY = (copy_elems + GQA_THREADS - 1) / GQA_THREADS;
  expand_kv_kernel<<<dim3(total_heads, blocksY), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst),
      heads_per_group, src_stride, dst_stride, copy_elems);
  return hipSuccess;
}

extern "C" int hip_gqa_causal_mask(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  causal_mask_kernel<<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<__half*>(S),
      skv, sq, batch_stride, past_len);
  return hipSuccess;
}

extern "C" int hip_gqa_softmax_inplace(
    void* stream_ptr, void* data,
    int total_head_queries, int rows, int cols,
    int batch_stride) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int threads = 32;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  softmax_inplace_kernel<<<total_head_queries, threads,
                           threads * sizeof(float), stream>>>(
      static_cast<__half*>(data),
      rows, cols, batch_stride);
  return hipSuccess;
}
