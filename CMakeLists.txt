
cmake_minimum_required(VERSION 3.26)
project(
  MorphiZen
  VERSION 1.0.0
  LANGUAGES C CXX)
cmake_policy(SET CMP0115 NEW)
option(morphizen_ENABLE_UNIT_TEST "enable vaip unit test or not" ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extra_functions.cmake)
add_subdirectory (encryption)
add_subdirectory (vaip_core)
if(morphizen_ENABLE_UNIT_TEST)
  add_subdirectory (unit-test)
endif(morphizen_ENABLE_UNIT_TEST)
