/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

#include <cstdio>
#include <stdexcept>

// Stub implementation of com.microsoft.MultiHeadAttention.
//
// MultiHeadAttention is a heavy compute op (Q*K' + softmax + S*V) that
// today has no GPU implementation in this runtime. The compiler still
// emits a call into this symbol so that models containing the op can
// compile end-to-end and surface a clear runtime error instead of
// silently producing garbage.
//
// This function logs every parameter to stderr and then throws a
// std::runtime_error so the caller (the EP custom-op host) gets a clean
// abort path. Future work: route this to a real GPU implementation
// (likely sharing code with hip.gqa / hip.linear_attention).
int wrap_multi_head_attention(
    RuntimeState *state,
    // Inputs (10)
    void *query, void *key, void *value, void *bias, void *key_padding_mask,
    void *attention_bias, void *past_key, void *past_value,
    void *past_sequence_length, void *cache_indirection,
    // Outputs (4)
    void *output, void *present_key, void *present_value, void *qk,
    // Attributes (4)
    int64_t num_heads, float mask_filter_value, float scale,
    int64_t unidirectional,
    // Shape info (8)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t query_hidden, int64_t v_hidden, int64_t head_size,
    int64_t query_rank, int64_t element_size_bytes) {
  (void)state;
  (void)query;
  (void)output;

  std::fprintf(
      stderr,
      "[STUB] wrap_multi_head_attention(\n"
      "  query=%p, key=%s, value=%s, bias=%s,\n"
      "  key_padding_mask=%s, attention_bias=%s,\n"
      "  past_key=%s, past_value=%s,\n"
      "  past_sequence_length=%s, cache_indirection=%s,\n"
      "  output=%p, present_key=%s, present_value=%s, qk=%s,\n"
      "  num_heads=%lld, mask_filter_value=%f, scale=%f, "
      "unidirectional=%lld,\n"
      "  batch=%lld, seq_q=%lld, seq_kv=%lld, query_hidden=%lld, "
      "v_hidden=%lld, head_size=%lld, query_rank=%lld, "
      "element_size_bytes=%lld)\n",
      query, key ? "yes" : "null", value ? "yes" : "null",
      bias ? "yes" : "null", key_padding_mask ? "yes" : "null",
      attention_bias ? "yes" : "null", past_key ? "yes" : "null",
      past_value ? "yes" : "null",
      past_sequence_length ? "yes" : "null",
      cache_indirection ? "yes" : "null", output,
      present_key ? "yes" : "null", present_value ? "yes" : "null",
      qk ? "yes" : "null", (long long)num_heads, (double)mask_filter_value,
      (double)scale, (long long)unidirectional, (long long)batch_size,
      (long long)seq_len_q, (long long)seq_len_kv, (long long)query_hidden,
      (long long)v_hidden, (long long)head_size, (long long)query_rank,
      (long long)element_size_bytes);
  std::fflush(stderr);

  throw std::runtime_error(
      "wrap_multi_head_attention is not implemented (com.microsoft."
      "MultiHeadAttention has no GPU backend yet)");
}
