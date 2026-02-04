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

# HIP kernels library (compiled with hipcc)
add_subdirectory(kernels)

add_subdirectory(proto)
add_subdirectory(rocm-pass-common)
add_subdirectory(level-1-pass-rocm)

# Level-2 passes for different op types
add_subdirectory(level-2-pass-rocm-conv)
add_subdirectory(level-2-pass-rocm-gemm)
add_subdirectory(level-2-pass-rocm-matmul)
add_subdirectory(level-2-pass-rocm-mul)
add_subdirectory(level-2-pass-rocm-softmax)
add_subdirectory(level-2-pass-rocm-reshape)
add_subdirectory(level-2-pass-rocm-transpose)
add_subdirectory(level-2-pass-rocm-tile)

add_subdirectory(custom-op-rocm)

# Add test directory (test_gqa can be built independently)
add_subdirectory(test)
