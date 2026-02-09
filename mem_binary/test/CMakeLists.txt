##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(GTest REQUIRED)
find_package(ZLIB QUIET)
set(ENABLE_COMPRESSION "0")
if(TARGET ZLIB::ZLIB)
  set(ENABLE_COMPRESSION "1")
endif()

set(TEST_EXE_NAME test_mem_binary)

set(MEM_BINARY_CONTENT_FILE mem_binary_file.hpp.inc)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/scripts"
    $<TARGET_FILE:Python3::Interpreter>
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/compress_binary.py
    "${ENABLE_COMPRESSION}"
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_embedded_resource.txt")

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}" PROPERTIES GENERATED
                                                                      TRUE)

configure_file(../src/mem_binary.cpp
               "${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp" @ONLY)

add_executable(
  ${TEST_EXE_NAME}
  src/test_main.cpp src/test_case.cpp ../include/morphizen/mem_binary.hpp
  "${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}")

target_include_directories(${TEST_EXE_NAME}
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")

target_link_libraries(${TEST_EXE_NAME} PRIVATE GTest::gtest morphizen-utils
                                               glog::glog Microsoft.GSL::GSL)
if(TARGET ZLIB::ZLIB)
  target_link_libraries(${TEST_EXE_NAME} PRIVATE ZLIB::ZLIB)
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE ENABLE_COMPRESSION=1)
endif()

include(CTest)
enable_testing()
add_test(NAME ${TEST_EXE_NAME} COMMAND ${TEST_EXE_NAME})
