##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(
  onnx-hipdnn-ep
  VERSION 1.0.0
  LANGUAGES C CXX)

# Set MSVC runtime library to static to match pre-built dependencies
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # Silence MLIR std::complex<APFloat> deprecation warning
  add_compile_definitions(_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
add_subdirectory(proto)
add_subdirectory(level-1-pass-hipdnn)
add_subdirectory(custom-op-hipdnn)
add_subdirectory(test)
