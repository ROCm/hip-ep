
cmake_minimum_required(VERSION 3.27)
project(
  MorphiZen
  VERSION 1.0.0
  LANGUAGES C CXX)
cmake_policy(SET CMP0115 NEW)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extra_functions.cmake)
add_subdirectory (encryption)
add_subdirectory (vaip-core)
add_subdirectory (graph-opt)
add_subdirectory (vaip_pass_init)
if(morphizen_ENABLE_UNIT_TEST)
  include(CTest)
  enable_testing()
  add_subdirectory (unit-test)
endif()
