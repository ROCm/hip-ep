/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_qmoe(
    RuntimeState *state, const void *input, const void *router_probs,
    const void *fc1_weights, const void *fc1_scales, const void *fc1_bias,
    const void *fc2_weights, const void *fc2_scales, const void *fc2_bias,
    const void *fc3_weights, const void *fc3_scales, const void *fc3_bias,
    const void *fc1_zero_points, const void *fc2_zero_points,
    const void *fc3_zero_points, void *output, int64_t num_tokens,
    int64_t hidden_size, int64_t inter_size, int64_t num_experts, int64_t k,
    int64_t expert_weight_bits, int64_t block_size, int64_t swiglu_fusion,
    int64_t activation_type, float activation_alpha, float activation_beta,
    float swiglu_limit, int64_t normalize_routing_weights, int64_t elem_size);

class QMoETest : public RuntimeTestFixture {};

TEST_F(QMoETest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_qmoe(nullptr, dummy, dummy, dummy, dummy, nullptr, dummy,
                       dummy, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, dummy, 1, 64, 128, 4, 1, 4, 32, 0, 0,
                       0.0f, 0.0f, 0.0f, 0, 4),
            -1);
}

TEST_F(QMoETest, MinimalCallReturnsSuccess) {
  void *input = allocBuffer(1 * 64 * sizeof(float));
  void *routerProbs = allocBuffer(1 * 4 * sizeof(float));
  void *fc1Weights = allocBuffer(4 * 128 * 64 / 2);
  void *fc1Scales = allocBuffer(4 * 128 * 2 * sizeof(float));
  void *fc2Weights = allocBuffer(4 * 64 * 128 / 2);
  void *fc2Scales = allocBuffer(4 * 64 * 4 * sizeof(float));
  void *output = allocBuffer(1 * 64 * sizeof(float));

  EXPECT_EQ(wrap_qmoe(state, input, routerProbs, fc1Weights, fc1Scales,
                       nullptr, fc2Weights, fc2Scales, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, output, 1,
                       64, 128, 4, 1, 4, 32, 0, 0, 0.0f, 0.0f, 0.0f, 0, 4),
            0);
}

TEST_F(QMoETest, WithOptionalInputsReturnsSuccess) {
  void *input = allocBuffer(2 * 64 * sizeof(float));
  void *routerProbs = allocBuffer(2 * 8 * sizeof(float));
  void *fc1Weights = allocBuffer(8 * 256 * 64 / 2);
  void *fc1Scales = allocBuffer(8 * 256 * 2 * sizeof(float));
  void *fc1Bias = allocBuffer(8 * 256 * sizeof(float));
  void *fc2Weights = allocBuffer(8 * 64 * 256 / 2);
  void *fc2Scales = allocBuffer(8 * 64 * 8 * sizeof(float));
  void *fc2Bias = allocBuffer(8 * 64 * sizeof(float));
  void *fc3Weights = allocBuffer(8 * 256 * 64 / 2);
  void *fc3Scales = allocBuffer(8 * 256 * 2 * sizeof(float));
  void *output = allocBuffer(2 * 64 * sizeof(float));

  EXPECT_EQ(wrap_qmoe(state, input, routerProbs, fc1Weights, fc1Scales,
                       fc1Bias, fc2Weights, fc2Scales, fc2Bias, fc3Weights,
                       fc3Scales, nullptr, nullptr, nullptr, nullptr, output, 2,
                       64, 256, 8, 1, 4, 32, 1, 2, 0.0f, 0.0f, 0.0f, 1, 4),
            0);
}
