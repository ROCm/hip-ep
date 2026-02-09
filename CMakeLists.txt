##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.27)
project(
  morphizen
  VERSION 1.0.0
  LANGUAGES C CXX)
cmake_policy(SET CMP0115 NEW)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extra_functions.cmake)

# Add utility library first (other projects may depend on it)
add_subdirectory(morphizen-utils)
add_subdirectory(mem_binary)
add_subdirectory(encryption)
add_subdirectory(morphizen-ort-api-ext)
add_subdirectory(morphizen-graph)
add_subdirectory(morphizen-core)
if(morphizen_ENABLE_PATTERN_MATCHING)
  add_subdirectory(morphizen-pattern)
endif()
if(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  # when morphzen-core-dyanmic is enabled, we can build graph-opt, otherwise we will get link errors.
  if(morphizen_ENABLE_BOOST)
    # These executables depend on Boost library (primarily Boost::Process for subprocess execution).
    # They are only built when morphizen_ENABLE_BOOST is ON to allow building a lightweight
    # version of MorphiZen without Boost dependency when morphizen_ENABLE_BOOST is OFF.
    add_subdirectory(tools/graph-opt)
    add_subdirectory(tools/tar)
    add_subdirectory(tools/pattern-gen)
    add_subdirectory(tools/onnx-grep)
  endif()
endif(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)

add_subdirectory(morphizen-pass-init)
add_subdirectory(custom-op-generic)

if(morphizen_ENABLE_UNIT_TEST AND morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  # when morphzen-core-dyanmic is enabled, we can build unit-test, otherwise we will get link errors.
  include(CTest)
  enable_testing()
  add_subdirectory(unit-test)
  add_subdirectory(morphizen-demo)
endif()
if(morphizen_ENABLE_ORT_BRIDGE)
  if(morphizen_ENABLE_ONNX_BACKEND)
    add_subdirectory(onnx-ir-imp)
  endif()
  add_subdirectory(ort-bridge)
  if(morphizen_ENABLE_MLIR_BACKEND)
    add_subdirectory (mlir-imp)
  endif()
endif()
