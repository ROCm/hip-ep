/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_rotary_embedding(
    RuntimeState *state, void *input, void *position_ids, void *cos_cache,
    void *sin_cache, void *output, int64_t interleaved, int64_t num_heads,
    int64_t rotary_dim, int64_t input_num_elements,
    int64_t cos_cache_num_elements, int64_t element_size_bytes);

class RopeTest : public RuntimeTestFixture {};

TEST_F(RopeTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_rotary_embedding(nullptr, dummy, dummy, dummy, dummy, dummy, 0,
                                  8, 64, 1024, 512, 4),
            -1);
}

TEST_F(RopeTest, ValidCallReturnsSuccess) {
  void *input = allocBuffer(1 * 8 * 128 * sizeof(float));
  void *posIds = allocBuffer(1 * 8 * sizeof(int64_t));
  void *cosCache = allocBuffer(512 * 64 * sizeof(float));
  void *sinCache = allocBuffer(512 * 64 * sizeof(float));
  void *output = allocBuffer(1 * 8 * 128 * sizeof(float));

  EXPECT_EQ(wrap_rotary_embedding(state, input, posIds, cosCache, sinCache,
                                  output, 0, 8, 64, 1 * 8 * 128, 512 * 64, 4),
            0);
}

TEST_F(RopeTest, InterleavedModeReturnsSuccess) {
  void *input = allocBuffer(1024 * sizeof(float));
  void *posIds = allocBuffer(8 * sizeof(int64_t));
  void *cosCache = allocBuffer(256 * sizeof(float));
  void *sinCache = allocBuffer(256 * sizeof(float));
  void *output = allocBuffer(1024 * sizeof(float));

  EXPECT_EQ(wrap_rotary_embedding(state, input, posIds, cosCache, sinCache,
                                  output, 1, 4, 32, 1024, 256, 4),
            0);
}
