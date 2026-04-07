/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

extern "C" int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs,
                                   void *output, int64_t lhs_n, int64_t lhs_c,
                                   int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                                   int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                                   int64_t out_n, int64_t out_c, int64_t out_h,
                                   int64_t out_w, int64_t data_type,
                                   int64_t tensor_op);

extern "C" int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                                    void *output, int64_t num_elements,
                                    int64_t element_size_bytes);

class ElementwiseTest : public RuntimeTestFixture {};

TEST_F(ElementwiseTest, OpTensorNullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_miopenOpTensor(nullptr, dummy, dummy, dummy, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 0, 0),
            -1);
}

TEST_F(ElementwiseTest, OpTensorMulReturnsSuccess) {
  void *lhs = allocBuffer(2 * 3 * 4 * 4 * sizeof(float));
  void *rhs = allocBuffer(2 * 3 * 4 * 4 * sizeof(float));
  void *output = allocBuffer(2 * 3 * 4 * 4 * sizeof(float));

  // HIPDNN_EP_TENSOR_OP_MUL=0, HIPDNN_EP_DATATYPE_FLOAT=0
  EXPECT_EQ(
      wrap_miopenOpTensor(state, lhs, rhs, output, 2, 3, 4, 4, 2, 3, 4, 4, 2,
                          3, 4, 4, /*data_type=*/0, /*tensor_op=*/0),
      0);
}

TEST_F(ElementwiseTest, OpTensorAddReturnsSuccess) {
  void *lhs = allocBuffer(1 * 1 * 1 * 8 * sizeof(float));
  void *rhs = allocBuffer(1 * 1 * 1 * 8 * sizeof(float));
  void *output = allocBuffer(1 * 1 * 1 * 8 * sizeof(float));

  // HIPDNN_EP_TENSOR_OP_ADD=1
  EXPECT_EQ(
      wrap_miopenOpTensor(state, lhs, rhs, output, 1, 1, 1, 8, 1, 1, 1, 8, 1,
                          1, 1, 8, /*data_type=*/0, /*tensor_op=*/1),
      0);
}

TEST_F(ElementwiseTest, OpTensorBroadcastReturnsSuccess) {
  void *lhs = allocBuffer(1 * 128 * 32 * 1 * sizeof(float));
  void *rhs = allocBuffer(1 * 1 * 1 * 1 * sizeof(float));  // scalar broadcast
  void *output = allocBuffer(1 * 128 * 32 * 1 * sizeof(float));

  EXPECT_EQ(
      wrap_miopenOpTensor(state, lhs, rhs, output, 1, 128, 32, 1, 1, 1, 1, 1,
                          1, 128, 32, 1, /*data_type=*/0, /*tensor_op=*/1),
      0);
}

TEST_F(ElementwiseTest, SubNullStateReturnsError) {
  float dummy[1] = {0};
  EXPECT_EQ(wrap_elementwise_sub(nullptr, dummy, dummy, dummy, 10, 4), -1);
}

TEST_F(ElementwiseTest, SubValidCallReturnsSuccess) {
  void *lhs = allocBuffer(100 * sizeof(float));
  void *rhs = allocBuffer(100 * sizeof(float));
  void *output = allocBuffer(100 * sizeof(float));

  EXPECT_EQ(wrap_elementwise_sub(state, lhs, rhs, output, 100, 4), 0);
}
