/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_hipblasLtMatmul(RuntimeState *state, const void *A,
                                    const void *B, void *output, int64_t M,
                                    int64_t N, int64_t K, int64_t batch_count,
                                    int64_t elem_size);

class MatmulTest : public RuntimeTestFixture {};

TEST_F(MatmulTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_hipblasLtMatmul(nullptr, dummy, dummy, dummy, 4, 4, 4, 1, 4),
            -1);
}

TEST_F(MatmulTest, ValidCallReturnsSuccess) {
  void *A = allocBuffer(4 * 8 * sizeof(float));
  void *B = allocBuffer(8 * 16 * sizeof(float));
  void *output = allocBuffer(4 * 16 * sizeof(float));

  EXPECT_EQ(wrap_hipblasLtMatmul(state, A, B, output, 4, 16, 8, 1, 4), 0);
}

TEST_F(MatmulTest, BatchedCallReturnsSuccess) {
  void *A = allocBuffer(2 * 4 * 8 * sizeof(float));
  void *B = allocBuffer(2 * 8 * 16 * sizeof(float));
  void *output = allocBuffer(2 * 4 * 16 * sizeof(float));

  EXPECT_EQ(wrap_hipblasLtMatmul(state, A, B, output, 4, 16, 8, 2, 4), 0);
}
