/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

// Layout helpers
//
// Two input layouts are supported (selected by the is_bnsh flag):
//   is_bnsh == 0 : BSNH = [batch, seq_len, num_heads, head_dim]
//                  (also covers 3D [batch, seq_len, num_heads*head_dim] which
//                   has identical row-major strides)
//   is_bnsh != 0 : BNSH = [batch, num_heads, seq_len, head_dim]
//                  (ONNX com.microsoft.RotaryEmbedding 4D default; also the
//                   layout used for K/V in GQA-style attention)
//
// position_ids is always [batch, seq_len] regardless of input layout.

static __device__ __forceinline__ int rope_decompose_idx(
    int idx, int seq_len, int num_heads, int head_dim, int is_bnsh,
    int& b, int& s, int& h, int& d) {
  d = idx % head_dim;
  int r1 = idx / head_dim;
  if (is_bnsh) {
    s = r1 % seq_len;
    int r2 = r1 / seq_len;
    h = r2 % num_heads;
    b = r2 / num_heads;
  } else {
    h = r1 % num_heads;
    int r2 = r1 / num_heads;
    s = r2 % seq_len;
    b = r2 / seq_len;
  }
  return 0;
}

static __device__ __forceinline__ int rope_base_offset(
    int b, int s, int h, int seq_len, int num_heads, int head_dim,
    int is_bnsh) {
  if (is_bnsh) {
    return b * (num_heads * seq_len * head_dim) +
           h * (seq_len * head_dim) +
           s * head_dim;
  }
  return b * (seq_len * num_heads * head_dim) +
         s * (num_heads * head_dim) +
         h * head_dim;
}

// Resolve the cos/sin cache row index for token (b, s).
//
// Two cache forms are supported:
//   position_ids != null : 2D lookup table [max_seq_len, half]. The row is
//                          position_ids[b, s], clamped to the table bounds.
//   position_ids == null : cos/sin are already position-expanded to
//                          [batch, seq, half] (native ai.onnx RotaryEmbedding
//                          without position_ids). Element (b, s, *) lives at
//                          flat offset (b*seq+s)*half, so the row index is
//                          simply b*seq+s and no clamp is needed (it is always
//                          < batch*seq = the number of rows).
static __device__ __forceinline__ int rope_resolve_pos(
    const int64_t* position_ids, int b, int s, int seq_len, int max_seq_len) {
  if (!position_ids)
    return b * seq_len + s;
  int64_t pos_raw = position_ids[b * seq_len + s];
  return static_cast<int>(pos_raw < 0 ? 0 :
             (pos_raw >= max_seq_len ? max_seq_len - 1 : pos_raw));
}

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
    int max_seq_len,
    int is_bnsh) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seq_len * num_heads * head_dim;

  if (idx >= total_elements)
    return;

  int b, s, h, d;
  rope_decompose_idx(idx, seq_len, num_heads, head_dim, is_bnsh, b, s, h, d);

  if (d >= rotary_dim) {
    output[idx] = input[idx];
    return;
  }

  int pos = rope_resolve_pos(position_ids, b, s, seq_len, max_seq_len);

  if (d < half_rot) {
    T cos_val = cos_cache[pos * half_rot + d];
    T sin_val = sin_cache[pos * half_rot + d];

    int base = rope_base_offset(b, s, h, seq_len, num_heads, head_dim, is_bnsh);
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
    int max_seq_len,
    int is_bnsh) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = batch_size * seq_len * num_heads * half_rot;

  if (idx >= total_pairs)
    return;

  int pair = idx % half_rot;
  int r1 = idx / half_rot;
  int b, s, h;
  if (is_bnsh) {
    s = r1 % seq_len;
    int r2 = r1 / seq_len;
    h = r2 % num_heads;
    b = r2 / num_heads;
  } else {
    h = r1 % num_heads;
    int r2 = r1 / num_heads;
    s = r2 % seq_len;
    b = r2 / seq_len;
  }

  int pos = rope_resolve_pos(position_ids, b, s, seq_len, max_seq_len);

  T cos_val = cos_cache[pos * half_rot + pair];
  T sin_val = sin_cache[pos * half_rot + pair];

  int base = rope_base_offset(b, s, h, seq_len, num_heads, head_dim, is_bnsh);
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
    int rotary_dim,
    int is_bnsh) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int remaining_per_head = head_dim - rotary_dim;
  int total = batch_size * seq_len * num_heads * remaining_per_head;

  if (idx >= total || remaining_per_head <= 0)
    return;

  int local_d = idx % remaining_per_head;
  int r1 = idx / remaining_per_head;
  int b, s, h;
  if (is_bnsh) {
    s = r1 % seq_len;
    int r2 = r1 / seq_len;
    h = r2 % num_heads;
    b = r2 / num_heads;
  } else {
    h = r1 % num_heads;
    int r2 = r1 / num_heads;
    s = r2 % seq_len;
    b = r2 / seq_len;
  }

  int base = rope_base_offset(b, s, h, seq_len, num_heads, head_dim, is_bnsh);
  int src_idx = base + rotary_dim + local_d;
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
    int64_t element_size_bytes,
    int64_t is_bnsh) {

  // position_ids may be null: native ai.onnx RotaryEmbedding without
  // position_ids ships cos/sin already position-expanded, and the kernels index
  // them by the flat token position instead of looking up position_ids.
  if (!input || !cos_cache || !sin_cache || !output) {
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
  int bnsh = is_bnsh ? 1 : 0;

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
          bs, sl, nh, hd, rd, hr, max_seq_len, bnsh);
    } else {
      int total = bs * sl * nh * hd;
      int grid_size = (total + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_half_rotated_kernel<__half>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len, bnsh);
    }

    if (interleaved && rd < hd) {
      int remaining_per_head = hd - rd;
      int total_remaining = bs * sl * nh * remaining_per_head;
      int grid_size = (total_remaining + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_copy_remaining_kernel<__half>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, out, bs, sl, nh, hd, rd, bnsh);
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
          bs, sl, nh, hd, rd, hr, max_seq_len, bnsh);
    } else {
      int total = bs * sl * nh * hd;
      int grid_size = (total + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_half_rotated_kernel<float>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, pos, cos_c, sin_c, out,
          bs, sl, nh, hd, rd, hr, max_seq_len, bnsh);
    }

    if (interleaved && rd < hd) {
      int remaining_per_head = hd - rd;
      int total_remaining = bs * sl * nh * remaining_per_head;
      int grid_size = (total_remaining + block_size - 1) / block_size;
      hipLaunchKernelGGL(rope_copy_remaining_kernel<float>,
          dim3(grid_size), dim3(block_size), 0, stream,
          in, out, bs, sl, nh, hd, rd, bnsh);
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

static __device__ __forceinline__ int mrope_stream(
    int col, int layout, int section_t, int section_h, int section_w) {
  if (layout == 0) {
    if (col < section_t) {
      return 0;
    }
    return col < section_t + section_h ? 1 : 2;
  }
  if (col % 3 == 1 && col < 3 * section_h) {
    return 1;
  }
  if (col % 3 == 2 && col < 3 * section_w) {
    return 2;
  }
  return 0;
}

// One thread per (token, head, j): j < half rotates the pair (j, j + half),
// j >= half copies channel rotary_dim + (j - half).
template <typename T>
__global__ void mrope_half_rotated_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ position_ids,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    T* __restrict__ output,
    int total,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int half_rot,
    int per_head,
    int max_position,
    int section_t,
    int section_h,
    int section_w,
    int layout,
    float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int j = idx % per_head;
  int token_head = idx / per_head;
  int64_t base = static_cast<int64_t>(token_head) * head_dim;

  if (j >= half_rot) {
    int64_t d = base + rotary_dim + (j - half_rot);
    output[d] = input[d];
    return;
  }

  int token = token_head / num_heads;
  int stream = mrope_stream(j, layout, section_t, section_h, section_w);
  int64_t pos =
      position_ids[static_cast<int64_t>(stream) * num_tokens + token];
  int64_t i0 = base + j;
  int64_t i1 = i0 + half_rot;
  if (pos < 0 || pos >= max_position) {
    output[i0] = input[i0];
    output[i1] = input[i1];
    return;
  }

  int64_t c = pos * half_rot + j;
  float cs = static_cast<float>(
      static_cast<T>(static_cast<float>(cos_cache[c]) * scale));
  float sn = static_cast<float>(
      static_cast<T>(static_cast<float>(sin_cache[c]) * scale));
  float x0 = static_cast<float>(input[i0]);
  float x1 = static_cast<float>(input[i1]);
  output[i0] = static_cast<T>(x0 * cs - x1 * sn);
  output[i1] = static_cast<T>(x1 * cs + x0 * sn);
}

extern "C" int hip_mrope_forward(
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
    int64_t max_position,
    int64_t section_t,
    int64_t section_h,
    int64_t section_w,
    int64_t mrope_layout,
    float scale,
    int64_t element_size_bytes) {
  if (!input || !position_ids || !cos_cache || !sin_cache || !output) {
    fprintf(stderr, "hip_mrope_forward: null pointer argument\n");
    return -1;
  }
  if (batch_size < 0 || seq_len < 0 || num_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim ||
      max_position < 0 ||
      section_t + section_h + section_w != rotary_dim / 2) {
    fprintf(stderr,
            "hip_mrope_forward: invalid shape batch=%lld seq=%lld heads=%lld "
            "head_dim=%lld rotary_dim=%lld section=[%lld,%lld,%lld]\n",
            (long long)batch_size, (long long)seq_len, (long long)num_heads,
            (long long)head_dim, (long long)rotary_dim, (long long)section_t,
            (long long)section_h, (long long)section_w);
    return -1;
  }

  const int64_t per_head = rotary_dim / 2 + (head_dim - rotary_dim);
  const int64_t num_tokens = batch_size * seq_len;
  const int64_t total = num_tokens * num_heads * per_head;
  if (total == 0) {
    return 0;
  }
  if (total > INT32_MAX || max_position > INT32_MAX) {
    fprintf(stderr, "hip_mrope_forward: problem too large (%lld threads)\n",
            (long long)total);
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();
  const int block_size = 256;
  const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
  auto launch = [&](auto type_tag) {
    using T = decltype(type_tag);
    hipLaunchKernelGGL(
        mrope_half_rotated_kernel<T>, dim3(grid_size), dim3(block_size), 0,
        stream, static_cast<const T*>(input),
        static_cast<const int64_t*>(position_ids),
        static_cast<const T*>(cos_cache), static_cast<const T*>(sin_cache),
        static_cast<T*>(output), static_cast<int>(total),
        static_cast<int>(num_tokens), static_cast<int>(num_heads),
        static_cast<int>(head_dim), static_cast<int>(rotary_dim),
        static_cast<int>(rotary_dim / 2), static_cast<int>(per_head),
        static_cast<int>(max_position), static_cast<int>(section_t),
        static_cast<int>(section_h), static_cast<int>(section_w),
        static_cast<int>(mrope_layout), scale);
  };
  if (element_size_bytes == 2) {
    launch(__half{});
  } else if (element_size_bytes == 4) {
    launch(float{});
  } else {
    fprintf(stderr, "hip_mrope_forward: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "hip_mrope_forward: kernel launch failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  return 0;
}
