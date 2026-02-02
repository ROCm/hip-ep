##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(
  morphizen-mlir
  VERSION 1.0.0
  LANGUAGES C CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
add_subdirectory(level-1-pass-mlir)
add_subdirectory(test)
