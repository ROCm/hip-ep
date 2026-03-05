/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "udna_custom_kernels.h"

// =============================================================================
// KV Cache Update Kernel
// =============================================================================
// Copies current key/value into the present_key/present_value buffer at the
// correct offset (after past sequence).
//
// ONNX com.microsoft.GroupQueryAttention uses BNSD layout for KV cache:
//   past_key:    [batch, kv_num_heads, past_seq, head_dim]       (BNSD)
//   present_key: [batch, kv_num_heads, total_seq, head_dim]      (BNSD)
//   key (current): [batch, seq_q, kv_num_heads * head_dim]       (BSD)
//
// The kernel copies past entries and new entries into the present buffer,
// converting current key/value from BSD to BNSD in the process.

template <typename T>
__global__ void kv_cache_update_kernel(
    const T* __restrict__ past,
    const T* __restrict__ current,
    T* __restrict__ present,
    const int32_t* __restrict__ seqlens_k,
    int64_t batch_size,
    int64_t past_seq_len,
    int64_t current_seq_len,
    int64_t total_seq_len,
    int64_t kv_num_heads,
    int64_t head_dim) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = batch_size * kv_num_heads * total_seq_len * head_dim;

  if (idx >= total)
    return;

  // Decompose flat index into BNSD coordinates
  int64_t d = idx % head_dim;
  int64_t remaining = idx / head_dim;
  int64_t s = remaining % total_seq_len;
  remaining = remaining / total_seq_len;
  int64_t n = remaining % kv_num_heads;
  int64_t b = remaining / kv_num_heads;

  int64_t actual_past_len = seqlens_k ? (int64_t)seqlens_k[b] : past_seq_len;

  // present is BNSD — idx already maps to the right position
  if (s < actual_past_len && past) {
    // past is BNSD: [batch, kv_num_heads, past_seq, head_dim]
    int64_t past_idx = ((b * kv_num_heads + n) * past_seq_len + s) * head_dim + d;
    present[idx] = past[past_idx];
  } else if (s >= actual_past_len && s < actual_past_len + current_seq_len) {
    // current is BSD: [batch, seq_q, kv_num_heads * head_dim]
    int64_t cur_s = s - actual_past_len;
    int64_t cur_idx = (b * current_seq_len + cur_s) * (kv_num_heads * head_dim)
                      + n * head_dim + d;
    present[idx] = current[cur_idx];
  }
}

// =============================================================================
// Attention Kernel (naive implementation)
// =============================================================================
// Computes scaled dot-product attention per (batch, query_head, query_pos).
//
// Key/value layout depends on kv_bnsd flag:
//   kv_bnsd=1: BNSD [batch, kv_num_heads, seq_kv, head_dim]  (present buffers)
//   kv_bnsd=0: BSD  [batch, seq_kv, kv_num_heads * head_dim] (direct key/value)
// Query is always BSD: [batch, seq_q, num_heads * head_dim]
//
// This is a straightforward O(N^2) implementation suitable for correctness
// verification and short sequences. For production use with long sequences,
// replace with a tiled/flash-attention approach or CK integration.

template <typename T>
__global__ void attention_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t num_heads,
    int64_t kv_num_heads,
    int64_t head_dim,
    float scale,
    float softcap,
    int64_t kv_bnsd) {

  // Each thread handles one (batch, head, query_pos) tuple
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = batch_size * num_heads * seq_len_q;

  if (idx >= total)
    return;

  int64_t q_pos = idx % seq_len_q;
  int64_t remaining = idx / seq_len_q;
  int64_t head = remaining % num_heads;
  int64_t b = remaining / num_heads;

  int64_t heads_per_kv = num_heads / kv_num_heads;
  int64_t kv_head = head / heads_per_kv;

  int64_t q_hidden = num_heads * head_dim;
  int64_t kv_hidden = kv_num_heads * head_dim;

  const T* q_row = query + b * seq_len_q * q_hidden +
                   q_pos * q_hidden + head * head_dim;

  // Precompute base offset for this batch+head in BNSD layout
  // BNSD: base = (b * kv_num_heads + kv_head) * seq_len_kv * head_dim
  // BSD:  base = b * seq_len_kv * kv_hidden, offset per pos = kv_head * head_dim
  int64_t kv_base_bnsd = (b * kv_num_heads + kv_head) * seq_len_kv * head_dim;
  int64_t kv_base_bsd = b * seq_len_kv * kv_hidden;

  // Step 1: Compute Q @ K^T scores and find max for numerical stability
  float max_score = -FLT_MAX;
  for (int64_t kv_pos = 0; kv_pos < seq_len_kv; kv_pos++) {
    const T* k_row = kv_bnsd
        ? key + kv_base_bnsd + kv_pos * head_dim
        : key + kv_base_bsd + kv_pos * kv_hidden + kv_head * head_dim;
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; d++) {
      dot += (float)q_row[d] * (float)k_row[d];
    }
    dot *= scale;

    if (softcap > 0.0f) {
      dot = softcap * tanhf(dot / softcap);
    }

    if (dot > max_score)
      max_score = dot;
  }

  // Step 2: Compute softmax denominator
  float sum_exp = 0.0f;
  for (int64_t kv_pos = 0; kv_pos < seq_len_kv; kv_pos++) {
    const T* k_row = kv_bnsd
        ? key + kv_base_bnsd + kv_pos * head_dim
        : key + kv_base_bsd + kv_pos * kv_hidden + kv_head * head_dim;
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; d++) {
      dot += (float)q_row[d] * (float)k_row[d];
    }
    dot *= scale;

    if (softcap > 0.0f) {
      dot = softcap * tanhf(dot / softcap);
    }

    sum_exp += expf(dot - max_score);
  }

  float inv_sum = 1.0f / (sum_exp + 1e-8f);

  // Step 3: Compute weighted sum of values
  T* out_row = output + b * seq_len_q * q_hidden +
               q_pos * q_hidden + head * head_dim;

  for (int64_t d = 0; d < head_dim; d++) {
    float acc = 0.0f;
    for (int64_t kv_pos = 0; kv_pos < seq_len_kv; kv_pos++) {
      const T* k_row = kv_bnsd
          ? key + kv_base_bnsd + kv_pos * head_dim
          : key + kv_base_bsd + kv_pos * kv_hidden + kv_head * head_dim;
      float dot = 0.0f;
      for (int64_t dd = 0; dd < head_dim; dd++) {
        dot += (float)q_row[dd] * (float)k_row[dd];
      }
      dot *= scale;

      if (softcap > 0.0f) {
        dot = softcap * tanhf(dot / softcap);
      }

      float weight = expf(dot - max_score) * inv_sum;
      const T* v_row = kv_bnsd
          ? value + kv_base_bnsd + kv_pos * head_dim
          : value + kv_base_bsd + kv_pos * kv_hidden + kv_head * head_dim;
      acc += weight * (float)v_row[d];
    }
    out_row[d] = (T)acc;
  }
}

// =============================================================================
// RoPE helper for GQA-internal rotary embedding
// =============================================================================

template <typename T>
__global__ void gqa_rope_kernel(
    T* __restrict__ data,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t interleaved,
    int64_t past_seq_offset) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t half_rot = rotary_dim / 2;

  if (interleaved) {
    int64_t total_pairs = batch_size * seq_len * num_heads * half_rot;
    if (idx >= total_pairs)
      return;

    int64_t pair = idx % half_rot;
    int64_t rem = idx / half_rot;
    int64_t h = rem % num_heads;
    rem = rem / num_heads;
    int64_t s = rem % seq_len;
    int64_t b = rem / seq_len;

    int64_t pos = past_seq_offset + s;
    T cos_val = cos_cache[pos * half_rot + pair];
    T sin_val = sin_cache[pos * half_rot + pair];

    int64_t hidden = num_heads * head_dim;
    int64_t base = b * seq_len * hidden + s * hidden + h * head_dim;
    int64_t i0 = base + 2 * pair;
    int64_t i1 = base + 2 * pair + 1;

    T x0 = data[i0];
    T x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
  } else {
    int64_t total = batch_size * seq_len * num_heads * half_rot;
    if (idx >= total)
      return;

    int64_t d = idx % half_rot;
    int64_t rem = idx / half_rot;
    int64_t h = rem % num_heads;
    rem = rem / num_heads;
    int64_t s = rem % seq_len;
    int64_t b = rem / seq_len;

    int64_t pos = past_seq_offset + s;
    T cos_val = cos_cache[pos * half_rot + d];
    T sin_val = sin_cache[pos * half_rot + d];

    int64_t hidden = num_heads * head_dim;
    int64_t base = b * seq_len * hidden + s * hidden + h * head_dim;
    int64_t i0 = base + d;
    int64_t i1 = base + d + half_rot;

    T x0 = data[i0];
    T x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
  }
}

// =============================================================================
// Host launcher
// =============================================================================

template <typename T>
static int gqa_forward_typed(
    hipStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const T* past_key,
    const T* past_value,
    const int32_t* seqlens_k,
    const void* total_seq_len_ptr,
    const T* cos_cache,
    const T* sin_cache,
    T* output,
    T* present_key,
    T* present_value,
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t num_heads,
    int64_t kv_num_heads,
    int64_t head_dim,
    float scale,
    float softcap,
    int64_t do_rotary,
    int64_t rotary_interleaved) {

  const int block_size = 256;
  int64_t kv_hidden = kv_num_heads * head_dim;
  int64_t past_seq_len = seq_len_kv - seq_len_q;
  if (past_seq_len < 0) past_seq_len = 0;

  // Step 1: KV cache update — build present_key/present_value
  if (present_key && present_value) {
    int64_t total_kv = batch_size * seq_len_kv * kv_hidden;
    int grid = (int)((total_kv + block_size - 1) / block_size);

    hipLaunchKernelGGL(kv_cache_update_kernel<T>,
        dim3(grid), dim3(block_size), 0, stream,
        past_key, key, present_key, seqlens_k,
        batch_size, past_seq_len, seq_len_q, seq_len_kv,
        kv_num_heads, head_dim);

    hipLaunchKernelGGL(kv_cache_update_kernel<T>,
        dim3(grid), dim3(block_size), 0, stream,
        past_value, value, present_value, seqlens_k,
        batch_size, past_seq_len, seq_len_q, seq_len_kv,
        kv_num_heads, head_dim);
  }

  // Use present_key/value for attention if available, otherwise use key/value.
  // Present buffers are BNSD [B,N,S,D], direct key/value are BSD [B,S,N*D].
  const T* att_key = present_key ? present_key : key;
  const T* att_value = present_value ? present_value : value;
  int64_t att_seq_kv = present_key ? seq_len_kv : seq_len_q;
  int64_t kv_bnsd = present_key ? 1 : 0;

  // Step 2: Optional RoPE on query and key
  if (do_rotary && cos_cache && sin_cache) {
    int64_t q_hidden = num_heads * head_dim;
    int64_t rotary_dim = head_dim;
    int64_t half_rot = rotary_dim / 2;

    // RoPE on query (copy to output, apply RoPE there)
    {
      int64_t total = batch_size * seq_len_q * num_heads * half_rot;
      int grid = (int)((total + block_size - 1) / block_size);
      hipMemcpyAsync((void*)output, (const void*)query,
                     batch_size * seq_len_q * q_hidden * sizeof(T),
                     hipMemcpyDeviceToDevice, stream);
      hipLaunchKernelGGL(gqa_rope_kernel<T>,
          dim3(grid), dim3(block_size), 0, stream,
          output, cos_cache, sin_cache,
          batch_size, seq_len_q, num_heads, head_dim,
          rotary_dim, rotary_interleaved, past_seq_len);
    }

    // TODO: RoPE on key within present_key needs BNSD-aware indexing.
    // The gqa_rope_kernel currently uses BSD layout. For do_rotary=1
    // with BNSD present buffers, this needs to be updated.
    if (present_key) {
      int64_t total = batch_size * seq_len_q * kv_num_heads * half_rot;
      int grid = (int)((total + block_size - 1) / block_size);
      T* new_key_start = present_key +
          batch_size * past_seq_len * kv_hidden;
      hipLaunchKernelGGL(gqa_rope_kernel<T>,
          dim3(grid), dim3(block_size), 0, stream,
          new_key_start, cos_cache, sin_cache,
          batch_size, seq_len_q, kv_num_heads, head_dim,
          rotary_dim, rotary_interleaved, past_seq_len);
    }

    // Attention using RoPE'd query (stored in output) and present KV
    {
      int64_t total_heads = batch_size * num_heads * seq_len_q;
      int grid = (int)((total_heads + block_size - 1) / block_size);

      T* q_rotated = nullptr;
      hipMalloc(&q_rotated, batch_size * seq_len_q * q_hidden * sizeof(T));
      hipMemcpyAsync(q_rotated, output,
                     batch_size * seq_len_q * q_hidden * sizeof(T),
                     hipMemcpyDeviceToDevice, stream);

      hipLaunchKernelGGL(attention_kernel<T>,
          dim3(grid), dim3(block_size), 0, stream,
          q_rotated, att_key, att_value, output,
          batch_size, seq_len_q, att_seq_kv,
          num_heads, kv_num_heads, head_dim,
          scale, softcap, kv_bnsd);

      hipStreamSynchronize(stream);
      hipFree(q_rotated);
    }
  } else {
    // Attention without RoPE
    int64_t total_heads = batch_size * num_heads * seq_len_q;
    int grid = (int)((total_heads + block_size - 1) / block_size);

    hipLaunchKernelGGL(attention_kernel<T>,
        dim3(grid), dim3(block_size), 0, stream,
        query, att_key, att_value, output,
        batch_size, seq_len_q, att_seq_kv,
        num_heads, kv_num_heads, head_dim,
        scale, softcap, kv_bnsd);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "udna_gqa_forward: kernel error: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  return 0;
}

extern "C" int udna_gqa_forward(
    void* stream_ptr,
    const void* query,
    const void* key,
    const void* value,
    const void* past_key,
    const void* past_value,
    const void* seqlens_k,
    const void* total_seq_len,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    void* present_key,
    void* present_value,
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t num_heads,
    int64_t kv_num_heads,
    int64_t head_dim,
    float scale,
    float softcap,
    int64_t do_rotary,
    int64_t rotary_interleaved,
    int64_t element_size_bytes) {

  if (!query || !key || !value || !output) {
    fprintf(stderr, "udna_gqa_forward: null pointer for required argument\n");
    return -1;
  }

  if (num_heads % kv_num_heads != 0) {
    fprintf(stderr, "udna_gqa_forward: num_heads (%lld) must be divisible by kv_num_heads (%lld)\n",
            (long long)num_heads, (long long)kv_num_heads);
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  if (element_size_bytes == 2) {
    return gqa_forward_typed<__half>(
        stream,
        static_cast<const __half*>(query),
        static_cast<const __half*>(key),
        static_cast<const __half*>(value),
        static_cast<const __half*>(past_key),
        static_cast<const __half*>(past_value),
        static_cast<const int32_t*>(seqlens_k),
        total_seq_len,
        static_cast<const __half*>(cos_cache),
        static_cast<const __half*>(sin_cache),
        static_cast<__half*>(output),
        static_cast<__half*>(present_key),
        static_cast<__half*>(present_value),
        batch_size, seq_len_q, seq_len_kv,
        num_heads, kv_num_heads, head_dim,
        scale, softcap, do_rotary, rotary_interleaved);
  } else if (element_size_bytes == 4) {
    return gqa_forward_typed<float>(
        stream,
        static_cast<const float*>(query),
        static_cast<const float*>(key),
        static_cast<const float*>(value),
        static_cast<const float*>(past_key),
        static_cast<const float*>(past_value),
        static_cast<const int32_t*>(seqlens_k),
        total_seq_len,
        static_cast<const float*>(cos_cache),
        static_cast<const float*>(sin_cache),
        static_cast<float*>(output),
        static_cast<float*>(present_key),
        static_cast<float*>(present_value),
        batch_size, seq_len_q, seq_len_kv,
        num_heads, kv_num_heads, head_dim,
        scale, softcap, do_rotary, rotary_interleaved);
  } else {
    fprintf(stderr, "udna_gqa_forward: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }
}
