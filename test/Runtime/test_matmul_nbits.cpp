/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_matmul_nbits(RuntimeState *state, const void *A,
                                 const void *B, const void *scales,
                                 const void *zero_points, const void *g_idx,
                                 const void *bias, void *output, int64_t M,
                                 int64_t N, int64_t K, int64_t batch_count,
                                 int64_t bits, int64_t block_size,
                                 int64_t elem_size);

class MatMulNBitsTest : public RuntimeTestFixture {};

TEST_F(MatMulNBitsTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_matmul_nbits(nullptr, dummy, dummy, dummy, nullptr, nullptr,
                              nullptr, dummy, 1, 1, 1, 1, 4, 32, 4),
            -1);
}

TEST_F(MatMulNBitsTest, ValidCallReturnsSuccess) {
  void *A = allocBuffer(4 * 256 * sizeof(float));
  void *B = allocBuffer(128 * 256 / 2); // 4-bit packed
  void *scales = allocBuffer(128 * (256 / 32) * sizeof(float));
  void *output = allocBuffer(4 * 128 * sizeof(float));

  EXPECT_EQ(wrap_matmul_nbits(state, A, B, scales, nullptr, nullptr, nullptr,
                              output, 4, 128, 256, 1, 4, 32, 4),
            0);
}

TEST_F(MatMulNBitsTest, WithZeroPointsAndBiasReturnsSuccess) {
  void *A = allocBuffer(4 * 256 * sizeof(float));
  void *B = allocBuffer(128 * 256 / 2);
  void *scales = allocBuffer(128 * 8 * sizeof(float));
  void *zeroPoints = allocBuffer(128 * 8);
  void *bias = allocBuffer(128 * sizeof(float));
  void *output = allocBuffer(4 * 128 * sizeof(float));

  EXPECT_EQ(wrap_matmul_nbits(state, A, B, scales, zeroPoints, nullptr, bias,
                              output, 4, 128, 256, 1, 4, 32, 4),
            0);
}
