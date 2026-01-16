##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
vai_add_library(
  NAME
  pass_level1_rocm
  SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pass_main.cpp
  DEPS
  rocm_proto
)

# Find HIP for GPU device queries
find_package(hip REQUIRED)
target_link_libraries(pass_level1_rocm PRIVATE hip::device)
