/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_reduce_sum(RuntimeState *state, void *data, void *axes,
                               void *output, int64_t data_num_elements,
                               int64_t output_num_elements,
                               int64_t axes_num_elements,
                               int64_t element_size_bytes, int64_t keepdims,
                               int64_t noop_with_empty_axes);

class ReduceSumTest : public RuntimeTestFixture {};

TEST_F(ReduceSumTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_reduce_sum(nullptr, dummy, dummy, dummy, 10, 1, 1, 4, 1, 0),
            -1);
}

TEST_F(ReduceSumTest, ValidCallReturnsSuccess) {
  void *data = allocBuffer(24 * sizeof(float));
  int64_t axes_val[] = {1};
  void *output = allocBuffer(6 * sizeof(float));

  EXPECT_EQ(wrap_reduce_sum(state, data, axes_val, output, 24, 6, 1, 4, 1, 0),
            0);
}

TEST_F(ReduceSumTest, ReduceAllAxesReturnsSuccess) {
  void *data = allocBuffer(100 * sizeof(float));
  int64_t axes_vals[] = {0, 1, 2};
  void *output = allocBuffer(1 * sizeof(float));

  EXPECT_EQ(wrap_reduce_sum(state, data, axes_vals, output, 100, 1, 3, 4, 1, 0),
            0);
}

TEST_F(ReduceSumTest, EmptyAxesWithNoopReturnsSuccess) {
  void *data = allocBuffer(100 * sizeof(float));
  void *output = allocBuffer(100 * sizeof(float));

  // noop_with_empty_axes=1, axes_num_elements=0
  EXPECT_EQ(wrap_reduce_sum(state, data, nullptr, output, 100, 100, 0, 4, 1, 1),
            0);
}
