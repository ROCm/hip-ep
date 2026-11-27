/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_state_internal.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

// Picks the buffer one cache is updated in. When the output is the input
// buffer the update is done in place. A distinct output first receives a copy
// of the whole input cache: blocks this call does not write must still appear
// in the result. Partially overlapping buffers have no defined copy result and
// are rejected.
int resolve_cache_target(hipStream_t stream, void *cache, void *cache_out,
                         size_t bytes, void **target) {
  *target = cache;
  if (cache_out == cache)
    return 0;
  const uintptr_t src = reinterpret_cast<uintptr_t>(cache);
  const uintptr_t dst = reinterpret_cast<uintptr_t>(cache_out);
  if (src < dst + bytes && dst < src + bytes)
    return -1;
  // std::call_once is avoided: this file is also compiled to bitcode that the
  // JIT links without the C++ runtime library.
  static std::atomic_flag warned = ATOMIC_FLAG_INIT;
  if (!warned.test_and_set(std::memory_order_relaxed))
    fprintf(stderr,
            "wrap_paged_attention: cache outputs are distinct buffers "
            "from the cache inputs; copying the caches on every call. "
            "Bind each output to its input buffer to avoid the copy.\n");
  if (hipMemcpyAsync(cache_out, cache, bytes, hipMemcpyDeviceToDevice,
                     stream) != hipSuccess)
    return -1;
  *target = cache_out;
  return 0;
}

} // namespace

int wrap_paged_attention(RuntimeState *state, void *query, void *key,
                         void *value, void *key_cache, void *value_cache,
                         void *cumulative_sequence_length, void *past_seqlens,
                         void *block_table, void *cos_cache, void *sin_cache,
                         void *output, void *key_cache_out,
                         void *value_cache_out, int64_t num_heads,
                         int64_t kv_num_heads, int64_t do_rotary,
                         int64_t rotary_interleaved, int64_t local_window_size,
                         float scale, int64_t num_tokens, int64_t query_dim1,
                         int64_t element_size_bytes, int64_t batch,
                         int64_t num_blocks, int64_t block_size,
                         int64_t head_size, int64_t max_blocks) {
  (void)cos_cache;
  (void)sin_cache;
  (void)rotary_interleaved;
  (void)query_dim1;
  if (!state)
    return -1;

  if (element_size_bytes != 2 || do_rotary != 0 || local_window_size != -1)
    return -1;
  if (head_size <= 0)
    return -1;
  if (scale == 0.f)
    scale = 1.f / std::sqrt(static_cast<float>(head_size));

  const size_t cache_bytes =
      static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) *
      static_cast<size_t>(kv_num_heads) * static_cast<size_t>(head_size) *
      static_cast<size_t>(element_size_bytes);
  void *key_target = nullptr;
  void *value_target = nullptr;
  if (resolve_cache_target(state->stream, key_cache, key_cache_out, cache_bytes,
                           &key_target) != 0 ||
      resolve_cache_target(state->stream, value_cache, value_cache_out,
                           cache_bytes, &value_target) != 0)
    return -1;
  if (num_tokens == 0)
    return 0;
  const size_t scratch_bytes = hip_paged_attention_scratch_bytes(
      num_heads, kv_num_heads, num_tokens, head_size, batch, block_size,
      max_blocks);
  if (scratch_bytes == 0 ||
      hipdnn_ep_state_ensure_paged_attention_scratch(state, scratch_bytes) != 0)
    return -1;
  return hip_paged_attention(
      state->stream, query, key, value, key_target, value_target,
      cumulative_sequence_length, past_seqlens, block_table, output, num_heads,
      kv_num_heads, num_tokens, head_size, batch, num_blocks, block_size,
      max_blocks, scale, hipdnn_ep_state_get_paged_attention_scratch(state),
      scratch_bytes);
}
