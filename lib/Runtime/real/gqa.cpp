/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "../debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// =============================================================================
// GroupQueryAttention via custom HIP kernel
// =============================================================================
//
// The HipToLLVM lowering extracts shape info from the memref types and passes:
//   batch_size, seq_len_q, seq_len_kv, head_dim, element_size_bytes
// along with pointers and attributes (num_heads, kv_num_heads, scale, etc.)
//
// This wrapper extracts the HIP stream from RuntimeState and delegates to
// the custom kernel library (hip_custom_kernels.lib).
// =============================================================================

int wrap_group_query_attention(
    RuntimeState* state,
    void* query, void* key, void* value,
    void* past_key, void* past_value,
    void* seqlens_k, void* total_seq_len,
    void* cos_cache, void* sin_cache,
    void* output, void* present_key, void* present_value,
    int64_t num_heads, int64_t kv_num_heads,
    float scale, float softcap,
    int64_t do_rotary, int64_t rotary_interleaved,
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t head_dim, int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "wrap_group_query_attention: null state\n");
    return -1;
  }

  void* stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_group_query_attention: null stream\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_group_query_attention: batch=%lld, seq_q=%lld, "
                    "seq_kv=%lld, num_heads=%lld, kv_heads=%lld, head_dim=%lld, "
                    "scale=%f, do_rotary=%lld, elem_size=%lld\n",
                    (long long)batch_size, (long long)seq_len_q,
                    (long long)seq_len_kv, (long long)num_heads,
                    (long long)kv_num_heads, (long long)head_dim,
                    (double)scale, (long long)do_rotary,
                    (long long)element_size_bytes);

  int rc = hip_gqa_forward(
      stream,
      query, key, value,
      past_key, past_value,
      seqlens_k, total_seq_len,
      cos_cache, sin_cache,
      output, present_key, present_value,
      batch_size, seq_len_q, seq_len_kv,
      num_heads, kv_num_heads, head_dim,
      scale, softcap,
      do_rotary, rotary_interleaved,
      element_size_bytes);

  if (rc != 0) {
    fprintf(stderr, "wrap_group_query_attention: hip_gqa_forward failed (rc=%d)\n", rc);
  } else {
    RUNTIME_DEBUG_LOG("[REAL] wrap_group_query_attention: completed successfully\n");
  }

  return rc;
}
