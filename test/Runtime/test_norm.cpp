/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_miopenT5LayerNormForward(
    RuntimeState *state, void *input, void *scale, void *output,
    int64_t input_num_elements, int64_t scale_num_elements,
    int64_t element_size_bytes, int64_t axis, float epsilon,
    int64_t stash_type);

extern "C" int wrap_skip_simplified_layer_norm(
    RuntimeState *state, void *input, void *skip, void *gamma, void *bias,
    void *output, void *input_skip_bias_sum, int64_t input_num_elements,
    int64_t gamma_num_elements, int64_t element_size_bytes, float epsilon);

class NormTest : public RuntimeTestFixture {};

TEST_F(NormTest, T5LayerNormNullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenT5LayerNormForward(nullptr, dummy, dummy, dummy, 64, 8,
                                          4, -1, 1e-5f, 1),
            -1);
}

TEST_F(NormTest, T5LayerNormValidCallReturnsSuccess) {
  void *input = allocBuffer(2 * 8 * 64 * sizeof(float));  // [2,8,64]
  void *scale = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(2 * 8 * 64 * sizeof(float));

  EXPECT_EQ(wrap_miopenT5LayerNormForward(state, input, scale, output,
                                          2 * 8 * 64, 64, 4, -1, 1e-5f, 1),
            0);
}

TEST_F(NormTest, SkipLayerNormNullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_skip_simplified_layer_norm(nullptr, dummy, dummy, dummy,
                                            nullptr, dummy, nullptr, 64, 8, 4,
                                            1e-5f),
            -1);
}

TEST_F(NormTest, SkipLayerNormValidCallReturnsSuccess) {
  void *input = allocBuffer(2 * 8 * 64 * sizeof(float));
  void *skip = allocBuffer(2 * 8 * 64 * sizeof(float));
  void *gamma = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(2 * 8 * 64 * sizeof(float));

  EXPECT_EQ(wrap_skip_simplified_layer_norm(state, input, skip, gamma, nullptr,
                                            output, nullptr, 2 * 8 * 64, 64, 4,
                                            1e-5f),
            0);
}

TEST_F(NormTest, SkipLayerNormWithBiasReturnsSuccess) {
  void *input = allocBuffer(2 * 8 * 64 * sizeof(float));
  void *skip = allocBuffer(2 * 8 * 64 * sizeof(float));
  void *gamma = allocBuffer(64 * sizeof(float));
  void *bias = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(2 * 8 * 64 * sizeof(float));
  void *skipSum = allocBuffer(2 * 8 * 64 * sizeof(float));

  EXPECT_EQ(wrap_skip_simplified_layer_norm(state, input, skip, gamma, bias,
                                            output, skipSum, 2 * 8 * 64, 64, 4,
                                            1e-5f),
            0);
}
