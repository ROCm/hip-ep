/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

int wrap_dynamic_sparse_attention(
    RuntimeState *state, const void *query, const void *key, const void *value,
    const void *past_key, const void *past_value, const void *selected_indices,
    const void *selected_counts, const void *seqlens_k, void *output,
    void *present_key, void *present_value, int64_t batch, int64_t seq_len,
    int64_t num_heads, int64_t kv_num_heads, int64_t head_size,
    int64_t capacity, int64_t max_selected, float scale) {
  if (!state || !query || !key || !value || !past_key || !past_value ||
      !selected_counts || !seqlens_k || !output || !present_key ||
      !present_value) {
    fprintf(stderr, "wrap_dynamic_sparse_attention: null required argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_dynamic_sparse_attention: null stream\n");
    return -1;
  }

  OP_PROFILE(
      "dynamic_sparse_attention",
      [&] {
        char b[96];
        snprintf(b, sizeof(b),
                 "b=%lld,sq=%lld,h=%lld,g=%lld,d=%lld,cap=%lld,k=%lld",
                 (long long)batch, (long long)seq_len, (long long)num_heads,
                 (long long)kv_num_heads, (long long)head_size,
                 (long long)capacity, (long long)max_selected);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_dynamic_sparse_attention: B=%lld S=%lld N=%lld "
      "kvN=%lld D=%lld cap=%lld max_selected=%lld scale=%f "
      "shared_cache=%d\n",
      (long long)batch, (long long)seq_len, (long long)num_heads,
      (long long)kv_num_heads, (long long)head_size, (long long)capacity,
      (long long)max_selected, (double)scale, past_key == present_key ? 1 : 0);

  const size_t ws_bytes = hip_dynamic_sparse_attention_workspace_size(
      batch, seq_len, num_heads, kv_num_heads, head_size, max_selected);
  void *workspace = nullptr;
  if (ws_bytes > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, ws_bytes) != 0) {
      fprintf(stderr,
              "wrap_dynamic_sparse_attention: workspace of %zu bytes "
              "unavailable\n",
              ws_bytes);
      return -1;
    }
    workspace = hipdnn_ep_state_get_workspace(state);
  }

  int rc = hip_dynamic_sparse_attention(
      stream, query, key, value, past_key, past_value,
      static_cast<const int32_t *>(selected_indices),
      static_cast<const int32_t *>(selected_counts),
      static_cast<const int32_t *>(seqlens_k), output, present_key,
      present_value, workspace, ws_bytes, batch, seq_len, num_heads,
      kv_num_heads, head_size, capacity, max_selected, scale);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: hip_dynamic_sparse_attention "
            "failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}
