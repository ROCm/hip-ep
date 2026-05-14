/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

// Stub: full paged KV attention is not implemented yet. The MLIR pipeline
// lowers com.microsoft.PagedAttention to this symbol so graphs compile and
// link; returns success without touching GPU buffers.
int wrap_paged_attention(
    RuntimeState *state, void *query, void *key, void *value, void *key_cache,
    void *value_cache, void *cumulative_sequence_length, void *past_seqlens,
    void *block_table, void *cos_cache, void *sin_cache, void *output,
    void *key_cache_out, void *value_cache_out, int64_t num_heads,
    int64_t kv_num_heads, int64_t do_rotary, int64_t rotary_interleaved,
    int64_t local_window_size, float scale, float softcap, int64_t num_tokens,
    int64_t query_dim1, int64_t element_size_bytes) {
  (void)state;
  (void)query;
  (void)key;
  (void)value;
  (void)key_cache;
  (void)value_cache;
  (void)cumulative_sequence_length;
  (void)past_seqlens;
  (void)block_table;
  (void)cos_cache;
  (void)sin_cache;
  (void)output;
  (void)key_cache_out;
  (void)value_cache_out;
  (void)num_heads;
  (void)kv_num_heads;
  (void)do_rotary;
  (void)rotary_interleaved;
  (void)local_window_size;
  (void)scale;
  (void)softcap;
  (void)num_tokens;
  (void)query_dim1;
  (void)element_size_bytes;
  return 0;
}
