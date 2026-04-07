/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_gather(RuntimeState *state, void *data, void *indices,
                           void *output, int64_t axis,
                           int64_t data_num_elements,
                           int64_t output_num_elements,
                           int64_t element_size_bytes);

class GatherTest : public RuntimeTestFixture {};

TEST_F(GatherTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_gather(nullptr, dummy, dummy, dummy, 0, 10, 5, 4), -1);
}

TEST_F(GatherTest, ValidCallReturnsSuccess) {
  void *data = allocBuffer(100 * sizeof(float));
  void *indices = allocBuffer(10 * sizeof(int64_t));
  void *output = allocBuffer(10 * sizeof(float));

  EXPECT_EQ(wrap_gather(state, data, indices, output, 0, 100, 10, 4), 0);
}

TEST_F(GatherTest, NonZeroAxisReturnsSuccess) {
  void *data = allocBuffer(100 * sizeof(float));
  void *indices = allocBuffer(5 * sizeof(int64_t));
  void *output = allocBuffer(50 * sizeof(float));

  EXPECT_EQ(wrap_gather(state, data, indices, output, 1, 100, 50, 4), 0);
}
