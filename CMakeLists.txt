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

# loads the functions precompile_version_rc and add_version_rc_to_listed_targets
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/ryzenai_bin_metadata
)
include(binMetaData)
precompile_version_rc(3rd-party/ryzenai_bin_metadata)

# Add utility library first (other projects may depend on it)
add_subdirectory(morphizen-utils)

add_subdirectory (encryption)
add_subdirectory (vaip-core)
if(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  # when morphzen-core-dyanmic is enabled, we can build graph-opt, otherwise we will get link errors.
  add_subdirectory (graph-opt)
  add_subdirectory (tar)
  add_subdirectory (pattern-gen)
  add_subdirectory (onnx-grep)
endif(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)

add_subdirectory (vaip_pass_init)
add_subdirectory (custom-op-generic)

if(morphizen_ENABLE_UNIT_TEST AND morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  # when morphzen-core-dyanmic is enabled, we can build unit-test, otherwise we will get link errors.
  include(CTest)
  enable_testing()
  add_subdirectory (unit-test)
endif()
if(morphizen_ENABALE_ORT_BRIDGE)
  add_subdirectory(ort-bridge)
endif()
