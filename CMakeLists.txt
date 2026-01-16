##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(
  morphizen-rocm
  VERSION 1.0.0
  LANGUAGES C CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/presets.cmake)

add_subdirectory(proto)
add_subdirectory(level-1-pass-rocm)
add_subdirectory(level-2-pass-rocm-conv)
add_subdirectory(level-2-pass-rocm-gemm)
add_subdirectory(custom-op-rocm)

# Add test directory if unit tests are enabled
if(morphizen_ENABLE_UNIT_TEST)
  add_subdirectory(test)
endif()
