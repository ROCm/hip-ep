/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_miopenConvolutionForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group);

class ConvTest : public RuntimeTestFixture {};

TEST_F(ConvTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenConvolutionForward(nullptr, dummy, 1, 1, 1, 1, dummy, 1,
                                          nullptr, dummy, 1, 1, 1, 1, 1, 1, 0,
                                          0, 0, 0, 1, 1, 1),
            -1);
}

TEST_F(ConvTest, NullInputReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenConvolutionForward(state, nullptr, 1, 1, 1, 1, dummy, 1,
                                          nullptr, dummy, 1, 1, 1, 1, 1, 1, 0,
                                          0, 0, 0, 1, 1, 1),
            -1);
}

TEST_F(ConvTest, NullWeightsReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenConvolutionForward(state, dummy, 1, 1, 1, 1, nullptr, 1,
                                          nullptr, dummy, 1, 1, 1, 1, 1, 1, 0,
                                          0, 0, 0, 1, 1, 1),
            -1);
}

TEST_F(ConvTest, NullOutputReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenConvolutionForward(state, dummy, 1, 1, 1, 1, dummy, 1,
                                          nullptr, nullptr, 1, 1, 1, 1, 1, 1,
                                          0, 0, 0, 0, 1, 1, 1),
            -1);
}

TEST_F(ConvTest, ValidCallReturnsSuccess) {
  void *input = allocBuffer(1 * 3 * 4 * 4 * sizeof(float));
  void *weights = allocBuffer(8 * 3 * 3 * 3 * sizeof(float));
  void *output = allocBufferFilled(1 * 8 * 2 * 2 * sizeof(float), 0xFF);

  int rc = wrap_miopenConvolutionForward(state, input, 1, 3, 4, 4, weights, 8,
                                         nullptr, output, 2, 2, 3, 3, 1, 1, 0,
                                         0, 0, 0, 1, 1, 1);
  EXPECT_EQ(rc, 0);

  // Mock memsets output to zero
  float *out = (float *)output;
  for (int i = 0; i < 1 * 8 * 2 * 2; i++) {
    EXPECT_EQ(out[i], 0.0f);
  }
}

TEST_F(ConvTest, WithBiasReturnsSuccess) {
  void *input = allocBuffer(1 * 3 * 4 * 4 * sizeof(float));
  void *weights = allocBuffer(8 * 3 * 3 * 3 * sizeof(float));
  void *bias = allocBuffer(8 * sizeof(float));
  void *output = allocBuffer(1 * 8 * 2 * 2 * sizeof(float));

  int rc = wrap_miopenConvolutionForward(state, input, 1, 3, 4, 4, weights, 8,
                                         bias, output, 2, 2, 3, 3, 1, 1, 0, 0,
                                         0, 0, 1, 1, 1);
  EXPECT_EQ(rc, 0);
}
