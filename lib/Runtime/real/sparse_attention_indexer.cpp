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

int wrap_sparse_attention_indexer(
    RuntimeState *state, const void *query, const void *query_norm_weight,
    const void *key_norm_weight, const void *cos_cache, const void *sin_cache,
    const void *mask, const void *past_key, void *selected_indices,
    void *present_key, int64_t batch, int64_t seq_len, int64_t past_len,
    int64_t num_heads, int64_t head_size, int64_t rotary_width,
    int64_t rotary_rows, int64_t rotary_batched, int64_t compress_ratio,
    int64_t token_budget, int64_t element_size_bytes, int64_t past_batch,
    int64_t mask_batch, int64_t mask_len, int64_t cos_batch, int64_t sin_batch,
    int64_t sin_rows, float epsilon, float scale) {
  if (!state || !query || !query_norm_weight || !key_norm_weight ||
      !cos_cache || !sin_cache || (past_len > 0 && !past_key) ||
      !selected_indices || !present_key) {
    fprintf(stderr, "wrap_sparse_attention_indexer: null required argument\n");
    return -1;
  }
  if (past_batch != batch) {
    fprintf(stderr,
            "wrap_sparse_attention_indexer: past_key batch %lld must be "
            "B=%lld\n",
            (long long)past_batch, (long long)batch);
    return -1;
  }
  if (mask && (mask_batch != batch || mask_len != past_len + seq_len)) {
    fprintf(stderr,
            "wrap_sparse_attention_indexer: mask [%lld, %lld] must be "
            "[B=%lld, P+S=%lld]\n",
            (long long)mask_batch, (long long)mask_len, (long long)batch,
            (long long)(past_len + seq_len));
    return -1;
  }
  if (sin_rows != rotary_rows ||
      (rotary_batched && (cos_batch != batch || sin_batch != batch))) {
    fprintf(stderr,
            "wrap_sparse_attention_indexer: cos_cache (batch %lld, rows %lld) "
            "and sin_cache (batch %lld, rows %lld) must match, with batch "
            "B=%lld when rank 3\n",
            (long long)cos_batch, (long long)rotary_rows, (long long)sin_batch,
            (long long)sin_rows, (long long)batch);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_sparse_attention_indexer: null stream\n");
    return -1;
  }

  OP_PROFILE(
      "sparse_attention_indexer",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "b=%lld,sq=%lld,past=%lld,h=%lld,mask=%d",
                 (long long)batch, (long long)seq_len, (long long)past_len,
                 (long long)num_heads, mask ? 1 : 0);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_sparse_attention_indexer: B=%lld S=%lld P=%lld N=%lld "
      "D=%lld R=%lld X=%lld batched=%lld r=%lld budget=%lld elem=%lld "
      "eps=%g scale=%f mask=%d\n",
      (long long)batch, (long long)seq_len, (long long)past_len,
      (long long)num_heads, (long long)head_size, (long long)rotary_width,
      (long long)rotary_rows, (long long)rotary_batched,
      (long long)compress_ratio, (long long)token_budget,
      (long long)element_size_bytes, (double)epsilon, (double)scale,
      mask ? 1 : 0);

  const size_t ws_bytes = hip_sparse_attention_indexer_workspace_size(
      batch, seq_len, past_len, num_heads, head_size, compress_ratio);
  void *workspace = nullptr;
  if (ws_bytes > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, ws_bytes) != 0) {
      fprintf(stderr,
              "wrap_sparse_attention_indexer: workspace of %zu bytes "
              "unavailable\n",
              ws_bytes);
      return -1;
    }
    workspace = hipdnn_ep_state_get_workspace(state);
  }

  int rc = hip_sparse_attention_indexer(
      stream, query, query_norm_weight, key_norm_weight, cos_cache, sin_cache,
      static_cast<const int64_t *>(mask), past_key,
      static_cast<int32_t *>(selected_indices), present_key, workspace,
      ws_bytes, batch, seq_len, past_len, num_heads, head_size, rotary_width,
      rotary_rows, rotary_batched, compress_ratio, token_budget,
      static_cast<int>(element_size_bytes), epsilon, scale);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_sparse_attention_indexer: hip_sparse_attention_indexer "
            "failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}
