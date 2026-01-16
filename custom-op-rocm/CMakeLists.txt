##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
vai_add_custom_op(
  NAME
  rocm
  SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/custom_op.cpp
  DEPS
  rocm_proto
)

# Find ROCm libraries
find_package(hip REQUIRED)
find_package(miopen REQUIRED)
find_package(hipblaslt REQUIRED)

target_link_libraries(custom_op_rocm PRIVATE
  hip::device
  MIOpen
  hipblaslt::hipblaslt
)
