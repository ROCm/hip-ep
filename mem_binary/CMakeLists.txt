##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.22)
project(
  mem_binary
  VERSION 1.0.0
  LANGUAGES C CXX)

set(MEM_BINARY_CONTENT_FILE mem_binary_file.hpp.inc)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
    $<TARGET_FILE:Python3::Interpreter>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compress_binary.py
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
    "${VAIP_EMBEDDED_RESOURCE_PATH}")

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}" PROPERTIES GENERATED
                                                                      TRUE)

configure_file(src/mem_binary.cpp "${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp"
               @ONLY)

set(LIB_NAME mem_binary)
add_library(
  ${LIB_NAME} STATIC
  include/morphizen/mem_binary.hpp ${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}")
add_library(morphizen::${LIB_NAME} ALIAS ${LIB_NAME})

target_compile_options(${LIB_NAME} PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
target_link_libraries(
  ${LIB_NAME}
  PRIVATE morphizen-utils
  PUBLIC ZLIB::ZLIB glog::glog)
target_include_directories(${LIB_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(${morphizen_ENABLE_UNIT_TEST})
  add_subdirectory(test)
endif()
