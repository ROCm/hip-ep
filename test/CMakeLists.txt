##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
vai_add_test(
  NAME
  rocm_conv_test
  SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cpp
  DEPS
  custom_op_rocm
)

vai_add_test(
  NAME
  rocm_gemm_test
  SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm.cpp
  DEPS
  custom_op_rocm
)
