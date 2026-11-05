/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

int wrap_dynamic_sparse_attention(
    RuntimeState *state, const void *query, const void *key, const void *value,
    const void *past_key, const void *past_value, const void *selected_indices,
    const void *selected_counts, const void *seqlens_k, const void *gate,
    const void *q_norm_weight, const void *k_norm_weight,
    const void *position_ids, const void *cos_cache, const void *sin_cache,
    void *output, void *present_key, void *present_value, int64_t batch,
    int64_t seq_len, int64_t num_heads, int64_t kv_num_heads, int64_t head_size,
    int64_t capacity, int64_t max_selected, int64_t past_key_batch,
    int64_t past_key_heads, int64_t past_key_head_size,
    int64_t past_value_batch, int64_t past_value_heads,
    int64_t past_value_capacity, int64_t past_value_head_size,
    int64_t present_key_capacity, int64_t present_key_head_size,
    int64_t present_value_capacity, int64_t present_value_head_size,
    int64_t indices_rows, int64_t counts_rows, int64_t seqlens_len,
    int64_t key_batch, int64_t key_seq_len, int64_t key_width,
    int64_t value_batch, int64_t value_seq_len, int64_t value_width,
    int64_t gate_batch, int64_t gate_seq_len, int64_t gate_width,
    int64_t q_norm_len, int64_t k_norm_len, int64_t position_sections,
    int64_t position_batch, int64_t position_seq_len, int64_t cos_rows,
    int64_t cos_width, int64_t sin_rows, int64_t sin_width, int64_t section_t,
    int64_t section_h, int64_t section_w, int64_t mrope_layout, float scale,
    float norm_epsilon, const void *past_sequence_length) {
  const bool packed = !key && !value;
  if (!state || !query || (!packed && (!key || !value)) || !past_key ||
      !past_value || !selected_counts || !seqlens_k || !output ||
      !present_key || !present_value) {
    fprintf(stderr, "wrap_dynamic_sparse_attention: null required argument\n");
    return -1;
  }
  const int64_t kv_width = kv_num_heads * head_size;
  if (!packed && (key_batch != batch || key_seq_len != seq_len ||
                  key_width != kv_width || value_batch != batch ||
                  value_seq_len != seq_len || value_width != kv_width)) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: key [%lld, %lld, %lld] and value "
            "[%lld, %lld, %lld] must both be [B=%lld, S=%lld, %lld]\n",
            (long long)key_batch, (long long)key_seq_len, (long long)key_width,
            (long long)value_batch, (long long)value_seq_len,
            (long long)value_width, (long long)batch, (long long)seq_len,
            (long long)kv_width);
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
  if (gate && (gate_batch != batch || gate_seq_len != seq_len ||
               gate_width != num_heads * head_size)) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: gate [%lld, %lld, %lld] must be "
            "[B=%lld, S=%lld, %lld]\n",
            (long long)gate_batch, (long long)gate_seq_len,
            (long long)gate_width, (long long)batch, (long long)seq_len,
            (long long)(num_heads * head_size));
    return -1;
  }

  const int prologue_parts = (q_norm_weight != nullptr) +
                             (k_norm_weight != nullptr) +
                             (position_ids != nullptr) +
                             (cos_cache != nullptr) + (sin_cache != nullptr);
  HipDsaPrologue prologue{};
  if (prologue_parts != 0) {
    const int64_t half = section_t + section_h + section_w;
    if (prologue_parts != 5 || q_norm_len != head_size ||
        k_norm_len != head_size || position_sections != 3 ||
        position_batch != batch || position_seq_len != seq_len ||
        cos_rows != sin_rows || cos_width != half || sin_width != half) {
      fprintf(stderr,
              "wrap_dynamic_sparse_attention: prologue needs q/k norm weights "
              "[%lld] (got %lld, %lld), position_ids [3, B=%lld, S=%lld] (got "
              "[%lld, %lld, %lld]) and cos/sin [rows, %lld] (got [%lld, %lld], "
              "[%lld, %lld])\n",
              (long long)head_size, (long long)q_norm_len,
              (long long)k_norm_len, (long long)batch, (long long)seq_len,
              (long long)position_sections, (long long)position_batch,
              (long long)position_seq_len, (long long)half, (long long)cos_rows,
              (long long)cos_width, (long long)sin_rows, (long long)sin_width);
      return -1;
    }
    prologue.q_norm_weight = q_norm_weight;
    prologue.k_norm_weight = k_norm_weight;
    prologue.position_ids = static_cast<const int64_t *>(position_ids);
    prologue.cos_cache = cos_cache;
    prologue.sin_cache = sin_cache;
    prologue.rotary_rows = cos_rows;
    prologue.rotary_dim = 2 * half;
    prologue.section[0] = section_t;
    prologue.section[1] = section_h;
    prologue.section[2] = section_w;
    prologue.layout = mrope_layout;
    prologue.epsilon = norm_epsilon;
  }
  const HipDsaPrologue *prologue_ptr =
      prologue_parts != 0 ? &prologue : nullptr;

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_dynamic_sparse_attention: null stream\n");
    return -1;
  }

  HipDsaPair pair{};
  if (past_sequence_length) {
    if (!hipdnn_ep_read_past_sequence_length(
            state, stream, past_sequence_length, &pair.past_len) ||
        pair.past_len < 0) {
      fprintf(stderr, "wrap_dynamic_sparse_attention: failed to read the "
                      "paired past_sequence_length\n");
      return -1;
    }
    pair.gqa_policy = state->gqa_autotune_policy;
  }
  const HipDsaPair *pair_ptr = past_sequence_length ? &pair : nullptr;

  OP_PROFILE(
      "dynamic_sparse_attention",
      [&] {
        char b[128];
        snprintf(b, sizeof(b),
                 "b=%lld,sq=%lld,h=%lld,g=%lld,d=%lld,cap=%lld,k=%lld%s%s%s%s",
                 (long long)batch, (long long)seq_len, (long long)num_heads,
                 (long long)kv_num_heads, (long long)head_size,
                 (long long)capacity, (long long)max_selected,
                 packed ? ",packed" : "", prologue_ptr ? ",prologue" : "",
                 gate ? ",gate" : "", pair_ptr ? ",pair" : "");
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_dynamic_sparse_attention: B=%lld S=%lld N=%lld "
      "kvN=%lld D=%lld cap=%lld max_selected=%lld scale=%f "
      "shared_cache=%d packed=%d prologue=%d gate=%d pair_past=%lld\n",
      (long long)batch, (long long)seq_len, (long long)num_heads,
      (long long)kv_num_heads, (long long)head_size, (long long)capacity,
      (long long)max_selected, (double)scale, past_key == present_key ? 1 : 0,
      packed ? 1 : 0, prologue_ptr ? 1 : 0, gate ? 1 : 0,
      (long long)(pair_ptr ? pair.past_len : -1));

  const size_t ws_bytes = hip_dynamic_sparse_attention_workspace_size(
      batch, seq_len, num_heads, kv_num_heads, head_size, max_selected,
      prologue_ptr != nullptr);
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
      static_cast<const int32_t *>(seqlens_k), gate, prologue_ptr, output,
      present_key, present_value, workspace, ws_bytes, batch, seq_len,
      num_heads, kv_num_heads, head_size, capacity, max_selected, scale,
      pair_ptr);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_dynamic_sparse_attention: hip_dynamic_sparse_attention "
            "failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}
