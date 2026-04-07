/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_group_query_attention(
    RuntimeState *state, void *query, void *key, void *value, void *past_key,
    void *past_value, void *seqlens_k, void *total_seq_len, void *cos_cache,
    void *sin_cache, void *position_ids, void *attention_bias, void *head_sink,
    void *k_scale, void *v_scale, void *output, void *present_key,
    void *present_value, void *output_qk, int64_t num_heads,
    int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width, int64_t batch_size,
    int64_t seq_len_q, int64_t seq_len_kv, int64_t head_dim,
    int64_t element_size_bytes);

class GqaTest : public RuntimeTestFixture {};

TEST_F(GqaTest, NullStateReturnsError) {
  EXPECT_EQ(wrap_group_query_attention(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, 8, 2, 0.125f, 0, 0,
                0.0f, -1, 0, 0, 0, 0, 8, 1, 1, 128, 64, 4),
            -1);
}

TEST_F(GqaTest, MinimalValidCallReturnsSuccess) {
  // Minimal GQA: query + seqlens_k + total_seq_len + outputs
  void *query = allocBuffer(1 * 1 * 512 * sizeof(float));
  void *seqlensK = allocBuffer(1 * sizeof(int32_t));
  void *totalSeqLen = allocBuffer(1 * sizeof(int32_t));
  void *output = allocBuffer(1 * 1 * 512 * sizeof(float));
  void *presentKey = allocBuffer(1 * 2 * 128 * 64 * sizeof(float));
  void *presentValue = allocBuffer(1 * 2 * 128 * 64 * sizeof(float));

  EXPECT_EQ(wrap_group_query_attention(
                state, query, nullptr, nullptr, nullptr, nullptr, seqlensK,
                totalSeqLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, output, presentKey, presentValue, nullptr, 8,
                2, 0.125f, 0, 0, 0.0f, -1, 0, 0, 0, 0, 8, 1, 1, 128, 64, 4),
            0);
}

TEST_F(GqaTest, FullInputsCallReturnsSuccess) {
  void *query = allocBuffer(1 * 1 * 512 * sizeof(float));
  void *key = allocBuffer(1 * 1 * 128 * sizeof(float));
  void *value = allocBuffer(1 * 1 * 128 * sizeof(float));
  void *pastKey = allocBuffer(1 * 2 * 127 * 64 * sizeof(float));
  void *pastValue = allocBuffer(1 * 2 * 127 * 64 * sizeof(float));
  void *seqlensK = allocBuffer(1 * sizeof(int32_t));
  void *totalSeqLen = allocBuffer(1 * sizeof(int32_t));
  void *output = allocBuffer(1 * 1 * 512 * sizeof(float));
  void *presentKey = allocBuffer(1 * 2 * 128 * 64 * sizeof(float));
  void *presentValue = allocBuffer(1 * 2 * 128 * 64 * sizeof(float));

  EXPECT_EQ(wrap_group_query_attention(
                state, query, key, value, pastKey, pastValue, seqlensK,
                totalSeqLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, output, presentKey, presentValue, nullptr, 8,
                2, 0.125f, 0, 0, 0.0f, -1, 0, 0, 0, 0, 8, 1, 1, 128, 64, 4),
            0);
}
