/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

struct DeviceMeta {
  const int32_t *cumulative;
  const int32_t *past;
  const int32_t *block_table;
};

__device__ int find_batch(const int32_t *cumulative, int batch, int token) {
  for (int b = 0; b < batch; ++b) {
    if (token >= cumulative[b] && token < cumulative[b + 1])
      return b;
  }
  return -1;
}

__device__ int physical_slot(const int32_t *block_table, int max_blocks,
                             int block_size, int batch_index, int position) {
  if (position < 0)
    return -1;
  const int column = position / block_size;
  if (column >= max_blocks)
    return -1;
  const int block_id = block_table[batch_index * max_blocks + column];
  if (block_id < 0)
    return -1;
  return block_id * block_size + (position % block_size);
}

__global__ void write_paged_kv(const __half *key, const __half *value,
                               __half *key_cache, __half *value_cache,
                               DeviceMeta meta, int num_tokens, int batch,
                               int kv_heads, int head_size, int block_size,
                               int max_blocks, int cache_slots) {
  const int token = blockIdx.x;
  if (token >= num_tokens)
    return;
  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0)
    return;
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  const int slot =
      physical_slot(meta.block_table, max_blocks, block_size, batch_index, position);
  if (slot < 0 || slot >= cache_slots)
    return;

  const int width = kv_heads * head_size;
  const __half *key_row = key + static_cast<int64_t>(token) * width;
  const __half *value_row = value + static_cast<int64_t>(token) * width;
  __half *key_dst = key_cache + static_cast<int64_t>(slot) * width;
  __half *value_dst = value_cache + static_cast<int64_t>(slot) * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    key_dst[i] = key_row[i];
    value_dst[i] = value_row[i];
  }
}

constexpr int kAttentionThreads = 256;
constexpr int kMaxHeadSize = 512;
// Elements of a head a single lane owns when a wave is at least 32 lanes wide.
constexpr int kMaxDimsPerLane = kMaxHeadSize / 32;

__device__ float wave_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, warpSize);
  return value;
}

// One block per (query token, head). Each wave walks a strided subset of the
// causal context with an online softmax; the per-wave partials are merged
// through shared memory. A lane owns dims lane, lane + warpSize, ...
__global__ void __launch_bounds__(kAttentionThreads)
    paged_attention_fp16(const __half *query, const __half *key_cache,
                         const __half *value_cache, __half *output,
                         DeviceMeta meta, int num_tokens, int batch,
                         int num_heads, int kv_heads, int head_size,
                         int block_size, int max_blocks, int cache_slots,
                         float scale) {
  extern __shared__ float shared[];
  const int num_waves = blockDim.x / warpSize;
  float *wave_max = shared;
  float *wave_denom = shared + num_waves;
  float *wave_acc = shared + 2 * num_waves;

  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize;
  const int wave = tid / warpSize;
  __half *out =
      output + (static_cast<int64_t>(token) * num_heads + head) * head_size;

  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0) {
    for (int d = tid; d < head_size; d += blockDim.x)
      out[d] = __float2half(0.f);
    return;
  }
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  const int kv_head = head / (num_heads / kv_heads);
  const int kv_width = kv_heads * head_size;
  const __half *q =
      query + (static_cast<int64_t>(token) * num_heads + head) * head_size;

  float q_reg[kMaxDimsPerLane];
  float acc[kMaxDimsPerLane];
  for (int i = 0; i < kMaxDimsPerLane; ++i) {
    const int d = lane + i * warpSize;
    q_reg[i] = d < head_size ? __half2float(q[d]) : 0.f;
    acc[i] = 0.f;
  }

  float running_max = -INFINITY;
  float running_denom = 0.f;
  for (int past_pos = wave; past_pos <= position; past_pos += num_waves) {
    const int slot = physical_slot(meta.block_table, max_blocks, block_size,
                                   batch_index, past_pos);
    if (slot < 0 || slot >= cache_slots)
      continue;
    const int64_t row = static_cast<int64_t>(slot) * kv_width + kv_head * head_size;
    const __half *k = key_cache + row;
    const __half *v = value_cache + row;

    float partial = 0.f;
    for (int i = 0; i < kMaxDimsPerLane; ++i) {
      const int d = lane + i * warpSize;
      if (d < head_size)
        partial += q_reg[i] * __half2float(k[d]);
    }
    const float score = wave_sum(partial) * scale;

    const float new_max = fmaxf(running_max, score);
    const float correction =
        running_max == -INFINITY ? 0.f : expf(running_max - new_max);
    const float weight = expf(score - new_max);
    running_denom = running_denom * correction + weight;
    for (int i = 0; i < kMaxDimsPerLane; ++i) {
      const int d = lane + i * warpSize;
      if (d < head_size)
        acc[i] = acc[i] * correction + weight * __half2float(v[d]);
    }
    running_max = new_max;
  }

  if (lane == 0) {
    wave_max[wave] = running_max;
    wave_denom[wave] = running_denom;
  }
  for (int i = 0; i < kMaxDimsPerLane; ++i) {
    const int d = lane + i * warpSize;
    if (d < head_size)
      wave_acc[wave * head_size + d] = acc[i];
  }
  __syncthreads();

  float global_max = -INFINITY;
  for (int w = 0; w < num_waves; ++w)
    global_max = fmaxf(global_max, wave_max[w]);
  float denom = 0.f;
  if (global_max != -INFINITY) {
    for (int w = 0; w < num_waves; ++w) {
      if (wave_max[w] != -INFINITY)
        denom += wave_denom[w] * expf(wave_max[w] - global_max);
    }
  }
  for (int d = tid; d < head_size; d += blockDim.x) {
    float sum = 0.f;
    if (denom > 0.f) {
      for (int w = 0; w < num_waves; ++w) {
        if (wave_max[w] != -INFINITY)
          sum += wave_acc[w * head_size + d] * expf(wave_max[w] - global_max);
      }
      sum /= denom;
    }
    out[d] = __float2half(sum);
  }
}

int copy_i32(hipStream_t stream, const void *src, int count, int32_t **dst) {
  const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
  hipError_t err = hipMallocAsync(dst, bytes, stream);
  if (err != hipSuccess)
    return static_cast<int>(err);
  err = hipMemcpyAsync(*dst, src, bytes, hipMemcpyDefault, stream);
  return static_cast<int>(err);
}

}  // namespace

extern "C" int hip_paged_attention(
    void *stream_ptr, const void *query, const void *key, const void *value,
    void *key_cache, void *value_cache, const void *cumulative_sequence_length,
    const void *past_seqlens, const void *block_table, void *output,
    int64_t num_heads, int64_t kv_num_heads, int64_t num_tokens,
    int64_t head_size, int64_t batch, int64_t num_blocks, int64_t block_size,
    int64_t max_blocks, float scale) {
  if (num_tokens == 0)
    return 0;
  if (!query || !key || !value || !key_cache || !value_cache || !output ||
      !cumulative_sequence_length || !past_seqlens || !block_table)
    return -1;
  if (num_heads <= 0 || kv_num_heads <= 0 || head_size <= 0 || batch <= 0 ||
      num_blocks <= 0 || max_blocks <= 0 || block_size < 16 ||
      (block_size & (block_size - 1)) != 0 || num_heads % kv_num_heads != 0)
    return -1;
  const int64_t kIntMax = 2147483647;
  if (num_tokens > kIntMax || num_heads > kIntMax || kv_num_heads > kIntMax ||
      head_size > kIntMax || batch > kIntMax || num_blocks > kIntMax ||
      block_size > kIntMax || max_blocks > kIntMax ||
      num_blocks > kIntMax / block_size || batch > kIntMax / max_blocks)
    return -1;
  if (head_size > kMaxHeadSize || num_heads > 65535)
    return -1;
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_paged_attention tokens=%lld batch=%lld heads=%lld "
      "kv_heads=%lld head_size=%lld blocks=%lld block_size=%lld "
      "max_blocks=%lld\n",
      (long long)num_tokens, (long long)batch, (long long)num_heads,
      (long long)kv_num_heads, (long long)head_size, (long long)num_blocks,
      (long long)block_size, (long long)max_blocks);

  auto stream = static_cast<hipStream_t>(stream_ptr);
  int32_t *cumulative = nullptr;
  int32_t *past = nullptr;
  int32_t *table = nullptr;
  int status = copy_i32(stream, cumulative_sequence_length,
                        static_cast<int>(batch + 1), &cumulative);
  if (status == 0)
    status = copy_i32(stream, past_seqlens, static_cast<int>(batch), &past);
  if (status == 0)
    status = copy_i32(stream, block_table,
                      static_cast<int>(batch * max_blocks), &table);
  if (status != 0) {
    if (cumulative)
      (void)hipFreeAsync(cumulative, stream);
    if (past)
      (void)hipFreeAsync(past, stream);
    if (table)
      (void)hipFreeAsync(table, stream);
    return status;
  }

  DeviceMeta meta{cumulative, past, table};
  const int cache_slots = static_cast<int>(num_blocks * block_size);
  (void)hipGetLastError();
  hipLaunchKernelGGL(write_paged_kv, dim3(static_cast<unsigned>(num_tokens)),
                     dim3(128), 0, stream, static_cast<const __half *>(key),
                     static_cast<const __half *>(value),
                     static_cast<__half *>(key_cache),
                     static_cast<__half *>(value_cache), meta,
                     static_cast<int>(num_tokens), static_cast<int>(batch),
                     static_cast<int>(kv_num_heads), static_cast<int>(head_size),
                     static_cast<int>(block_size), static_cast<int>(max_blocks),
                     cache_slots);
  hipError_t err = hipGetLastError();
  if (err == hipSuccess) {
    int wave_size = 32;
    (void)hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, 0);
    const int num_waves = kAttentionThreads / wave_size;
    const size_t shared_bytes =
        (2 * static_cast<size_t>(num_waves) +
         static_cast<size_t>(num_waves) * static_cast<size_t>(head_size)) *
        sizeof(float);
    hipLaunchKernelGGL(
        paged_attention_fp16,
        dim3(static_cast<unsigned>(num_tokens), static_cast<unsigned>(num_heads)),
        dim3(kAttentionThreads), shared_bytes, stream,
        static_cast<const __half *>(query), static_cast<__half *>(key_cache),
        static_cast<__half *>(value_cache), static_cast<__half *>(output), meta,
        static_cast<int>(num_tokens), static_cast<int>(batch),
        static_cast<int>(num_heads), static_cast<int>(kv_num_heads),
        static_cast<int>(head_size), static_cast<int>(block_size),
        static_cast<int>(max_blocks), cache_slots, scale);
    err = hipGetLastError();
  }
  (void)hipFreeAsync(cumulative, stream);
  (void)hipFreeAsync(past, stream);
  (void)hipFreeAsync(table, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_paged_attention launch failed: %s\n",
            hipGetErrorString(err));
  }
  return err == hipSuccess ? 0 : static_cast<int>(err);
}
