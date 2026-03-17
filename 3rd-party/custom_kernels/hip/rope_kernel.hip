/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

// Half-rotated RoPE: dimensions [0..half) and [half..rotary_dim) form pairs.
// Uses int32 index arithmetic for RDNA GPU performance (int64 div/mod is slow).
template <typename T>
__global__ void rope_half_rotated_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ position_ids,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int half_rot,
    int max_seq_len) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seq_len * num_heads * head_dim;

  if (idx >= total_elements)
    return;

  int d = idx % head_dim;
  int remaining = idx / head_dim;
  int h = remaining % num_heads;
  remaining = remaining / num_heads;
  int s = remaining % seq_len;
  int b = remaining / seq_len;

  if (d >= rotary_dim) {
    output[idx] = input[idx];
    return;
  }

  int64_t pos_raw = position_ids[b * seq_len + s];
  int pos = static_cast<int>(pos_raw < 0 ? 0 :
                (pos_raw >= max_seq_len ? max_seq_len - 1 : pos_raw));

  if (d < half_rot) {
    T cos_val = cos_cache[pos * half_rot + d];
    T sin_val = sin_cache[pos * half_rot + d];

    int base = b * (seq_len * num_heads * head_dim) +
               s * (num_heads * head_dim) +
               h * head_dim;
    int idx_0 = base + d;
    int idx_1 = base + d + half_rot;

    T x0 = input[idx_0];
    T x1 = input[idx_1];
    output[idx_0] = x0 * cos_val - x1 * sin_val;
    output[idx_1] = x0 * sin_val + x1 * cos_val;
  }
}

// Interleaved RoPE: dimensions (2d, 2d+1) form pairs.
template <typename T>
__global__ void rope_interleaved_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ position_ids,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int half_rot,
    int max_seq_len) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = batch_size * seq_len * num_heads * half_rot;

  if (idx >= total_pairs)
    return;

  int pair = idx % half_rot;
  int remaining = idx / half_rot;
  int h = remaining % num_heads;
  remaining = remaining / num_heads;
  int s = remaining % seq_len;
  int b = remaining / seq_len;

  int64_t pos_raw = position_ids[b * seq_len + s];
  int pos = static_cast<int>(pos_raw < 0 ? 0 :
                (pos_raw >= max_seq_len ? max_seq_len - 1 : pos_raw));

  T cos_val = cos_cache[pos * half_rot + pair];
  T sin_val = sin_cache[pos * half_rot + pair];

  int base = b * (seq_len * num_heads * head_dim) +
             s * (num_heads * head_dim) +
             h * head_dim;
  int idx_0 = base + 2 * pair;
  int idx_1 = base + 2 * pair + 1;

  T x0 = input[idx_0];
  T x1 = input[idx_1];
  output[idx_0] = x0 * cos_val - x1 * sin_val;
  output[idx_1] = x0 * sin_val + x1 * cos_val;
}

// Copy kernel for dimensions beyond rotary_dim (interleaved mode).
template <typename T>
__global__ void rope_copy_remaining_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int remaining_per_head = head_dim - rotary_dim;
  int total = batch_size * seq_len * num_heads * remaining_per_head;

  if (idx >= total || remaining_per_head <= 0)
    return;

  int local_d = idx % remaining_per_head;
  int remaining = idx / remaining_per_head;
  int h = remaining % num_heads;
  remaining = remaining / num_heads;
  int s = remaining % seq_len;
  int b = remaining / seq_len;

  int src_idx = b * (seq_len * num_heads * head_dim) +
                s * (num_heads * head_dim) +
                h * head_dim + rotary_dim + local_d;
  output[src_idx] = input[src_idx];
}

extern "C" int hip_rope_forward(
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
    int64_t element_size_bytes) {

  if (!input || !position_ids || !cos_cache || !sin_cache || !output) {
    fprintf(stderr, "hip_rope_forward: null pointer argument\n");
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  const int block_size = 256;

  int bs = static_cast<int>(batch_size);
  int sl = static_cast<int>(seq_len);
  int nh = static_cast<int>(num_heads);
  int hd = static_cast<int>(head_dim);
  int rd = static_cast<int>(rotary_dim);
  int hr = rd / 2;
  int max_seq_len = static_cast<int>(max_seq_len_i64);

  if (element_size_bytes == 2) {
    auto* in = static_cast<const __half*>(input);
    auto* pos = static_cast<const int64_t*>(position_ids);
    auto* cos_c = static_cast<const __half*>(cos_cache);
    auto* sin_c = static_cast<const __half*>(sin_cache);
    auto* out = static_cast<__half*>(output);

    if (interleaved) {
      int total_pairs = bs * sl * nh * hr;
      int grid_size = (total_pairs + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_interleaved_kernel<__half>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len);
    } else {
      int total = bs * sl * nh * hd;
      int grid_size = (total + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_half_rotated_kernel<__half>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len);
    }

    if (interleaved && rd < hd) {
      int remaining_per_head = hd - rd;
      int total_remaining = bs * sl * nh * remaining_per_head;
      int grid_size = (total_remaining + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_copy_remaining_kernel<__half>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, out, bs, sl, nh, hd, rd);
    }
  } else if (element_size_bytes == 4) {
    auto* in = static_cast<const float*>(input);
    auto* pos = static_cast<const int64_t*>(position_ids);
    auto* cos_c = static_cast<const float*>(cos_cache);
    auto* sin_c = static_cast<const float*>(sin_cache);
    auto* out = static_cast<float*>(output);

    if (interleaved) {
      int total_pairs = bs * sl * nh * hr;
      int grid_size = (total_pairs + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_interleaved_kernel<float>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len);
    } else {
      int total = bs * sl * nh * hd;
      int grid_size = (total + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_half_rotated_kernel<float>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len);
    }

    if (interleaved && rd < hd) {
      int remaining_per_head = hd - rd;
      int total_remaining = bs * sl * nh * remaining_per_head;
      int grid_size = (total_remaining + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_copy_remaining_kernel<float>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, out, bs, sl, nh, hd, rd);
    }
  } else {
    fprintf(stderr, "hip_rope_forward: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "hip_rope_forward: kernel launch failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  return 0;
}
