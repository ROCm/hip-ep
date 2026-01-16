##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generate pattern include file from JSON
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_pattern_inc.cmake)

vai_add_library(
  NAME
  pass_level2_rocm_gemm
  SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pass_main.cpp
  DEPS
  rocm_proto
)

target_include_directories(pass_level2_rocm_gemm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
