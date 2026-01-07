##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(
  morphizen-hipdnn
  VERSION 1.0.0
  LANGUAGES C CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/presets.cmake)
add_subdirectory(proto)
add_subdirectory(level-1-pass-hipdnn)
add_subdirectory(custom-op-hipdnn)
add_subdirectory(test)
