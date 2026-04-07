/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_miopenActivationForward(RuntimeState *state, void *input,
                                            void *output, int64_t num_elements,
                                            int64_t data_type,
                                            int64_t activation_mode);

class ActivationTest : public RuntimeTestFixture {};

TEST_F(ActivationTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenActivationForward(nullptr, dummy, dummy, 1, 0, 0), -1);
}

TEST_F(ActivationTest, SigmoidReturnsSuccess) {
  void *input = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(64 * sizeof(float));

  // HIPDNN_EP_ACTIVATION_SIGMOID=0, HIPDNN_EP_DATATYPE_FLOAT=0
  EXPECT_EQ(wrap_miopenActivationForward(state, input, output, 64, 0, 0), 0);
}

TEST_F(ActivationTest, ReluReturnsSuccess) {
  void *input = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(64 * sizeof(float));

  // HIPDNN_EP_ACTIVATION_RELU=1
  EXPECT_EQ(wrap_miopenActivationForward(state, input, output, 64, 0, 1), 0);
}

TEST_F(ActivationTest, TanhReturnsSuccess) {
  void *input = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(64 * sizeof(float));

  // HIPDNN_EP_ACTIVATION_TANH=2
  EXPECT_EQ(wrap_miopenActivationForward(state, input, output, 64, 0, 2), 0);
}

TEST_F(ActivationTest, HalfPrecisionReturnsSuccess) {
  void *input = allocBuffer(64 * 2); // f16 = 2 bytes
  void *output = allocBuffer(64 * 2);

  // HIPDNN_EP_DATATYPE_HALF=1
  EXPECT_EQ(wrap_miopenActivationForward(state, input, output, 64, 1, 0), 0);
}
