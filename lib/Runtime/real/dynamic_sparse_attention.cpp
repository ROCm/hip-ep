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
    int64_t capacity, int64_t max_selected, int64_t past_key_batch,
    int64_t past_key_heads, int64_t past_key_head_size,
    int64_t past_value_batch, int64_t past_value_heads,
    int64_t past_value_capacity, int64_t past_value_head_size,
    int64_t present_key_capacity, int64_t present_key_head_size,
    int64_t present_value_capacity, int64_t present_value_head_size,
    int64_t indices_rows, int64_t counts_rows, int64_t seqlens_len,
    float scale) {
  if (!state || !query || !key || !value || !past_key || !past_value ||
      !selected_counts || !seqlens_k || !output || !present_key ||
      !present_value) {
    fprintf(stderr, "wrap_dynamic_sparse_attention: null required argument\n");
    return -1;
  }
  if (past_key_batch != batch || past_key_heads != kv_num_heads ||
      past_value_batch != batch || past_value_heads != kv_num_heads ||
      past_value_capacity != capacity) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: past_key [%lld, %lld, %lld] and "
            "past_value [%lld, %lld, %lld] must both be [B=%lld, "
            "kv_num_heads=%lld, capacity]\n",
            (long long)past_key_batch, (long long)past_key_heads,
            (long long)capacity, (long long)past_value_batch,
            (long long)past_value_heads, (long long)past_value_capacity,
            (long long)batch, (long long)kv_num_heads);
    return -1;
  }
  if (present_key_capacity != capacity || present_value_capacity != capacity) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: present capacity (%lld, %lld) "
            "must equal past capacity %lld\n",
            (long long)present_key_capacity, (long long)present_value_capacity,
            (long long)capacity);
    return -1;
  }
  if (past_key_head_size != head_size || past_value_head_size != head_size ||
      present_key_head_size != head_size ||
      present_value_head_size != head_size) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: cache head sizes (%lld, %lld, "
            "%lld, %lld) must equal query head size %lld\n",
            (long long)past_key_head_size, (long long)past_value_head_size,
            (long long)present_key_head_size,
            (long long)present_value_head_size, (long long)head_size);
    return -1;
  }
  if (indices_rows != batch * seq_len || counts_rows != batch * seq_len ||
      seqlens_len != batch) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: selected_indices rows %lld and "
            "selected_counts rows %lld must be B*S=%lld, seqlens_k length "
            "%lld must be B=%lld\n",
            (long long)indices_rows, (long long)counts_rows,
            (long long)(batch * seq_len), (long long)seqlens_len,
            (long long)batch);
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
