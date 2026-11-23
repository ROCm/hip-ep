/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_state_internal.h"

#include <cmath>

int wrap_paged_attention(RuntimeState *state, void *query, void *key,
                         void *value, void *key_cache, void *value_cache,
                         void *cumulative_sequence_length, void *past_seqlens,
                         void *block_table, void *cos_cache, void *sin_cache,
                         void *output, void *key_cache_out,
                         void *value_cache_out, int64_t num_heads,
                         int64_t kv_num_heads, int64_t do_rotary,
                         int64_t rotary_interleaved, int64_t local_window_size,
                         float scale, float softcap, int64_t num_tokens,
                         int64_t query_dim1, int64_t element_size_bytes,
                         int64_t batch, int64_t num_blocks, int64_t block_size,
                         int64_t head_size, int64_t max_blocks) {
  (void)cos_cache;
  (void)sin_cache;
  (void)rotary_interleaved;
  (void)query_dim1;
  if (!state)
    return -1;
  // Present aliases past. A distinct output buffer would drop the cache update.
  if ((key_cache_out && key_cache_out != key_cache) ||
      (value_cache_out && value_cache_out != value_cache))
    return -1;
  if (element_size_bytes != 2 || do_rotary != 0 || local_window_size != -1 ||
      softcap != 0.f)
    return -1;
  if (head_size <= 0)
    return -1;
  if (scale == 0.f)
    scale = 1.f / std::sqrt(static_cast<float>(head_size));
  if (num_tokens == 0)
    return 0;
  const size_t scratch_bytes = hip_paged_attention_scratch_bytes(
      num_heads, kv_num_heads, num_tokens, head_size, batch, block_size,
      max_blocks);
  if (scratch_bytes == 0 ||
      hipdnn_ep_state_ensure_paged_attention_scratch(state, scratch_bytes) != 0)
    return -1;
  return hip_paged_attention(
      state->stream, query, key, value, key_cache, value_cache,
      cumulative_sequence_length, past_seqlens, block_table, output, num_heads,
      kv_num_heads, num_tokens, head_size, batch, num_blocks, block_size,
      max_blocks, scale, hipdnn_ep_state_get_paged_attention_scratch(state),
      scratch_bytes);
}
