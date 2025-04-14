
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
if(morphizen_ENABLE_UNIT_TEST AND morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  # when morphzen-core-dyanmic is enabled, we can build unit-test, otherwise we will get link errors.
  include(CTest)
  enable_testing()
  add_subdirectory (unit-test)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_and_pack.cmake)
