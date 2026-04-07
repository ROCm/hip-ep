/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_cast(RuntimeState *state, void *input, void *output,
                         int64_t num_elements, int64_t src_data_type,
                         int64_t dst_data_type);

class CastTest : public RuntimeTestFixture {};

TEST_F(CastTest, NullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_cast(nullptr, dummy, dummy, 10, 0, 1), -1);
}

TEST_F(CastTest, Float32ToFloat16ReturnsSuccess) {
  void *input = allocBuffer(64 * sizeof(float));
  void *output = allocBuffer(64 * 2); // f16

  // HIPDNN_EP_DATATYPE_FLOAT=0, HIPDNN_EP_DATATYPE_HALF=1
  EXPECT_EQ(wrap_cast(state, input, output, 64, 0, 1), 0);
}

TEST_F(CastTest, Float16ToBFloat16ReturnsSuccess) {
  void *input = allocBuffer(64 * 2);
  void *output = allocBuffer(64 * 2);

  // HIPDNN_EP_DATATYPE_HALF=1, HIPDNN_EP_DATATYPE_BFLOAT16=2
  EXPECT_EQ(wrap_cast(state, input, output, 64, 1, 2), 0);
}

TEST_F(CastTest, Int32ToInt64ReturnsSuccess) {
  void *input = allocBuffer(32 * sizeof(int32_t));
  void *output = allocBuffer(32 * sizeof(int64_t));

  // HIPDNN_EP_DATATYPE_INT32=3, HIPDNN_EP_DATATYPE_INT64=4
  EXPECT_EQ(wrap_cast(state, input, output, 32, 3, 4), 0);
}
