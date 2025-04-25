#
# Copyright (C) 2022 Xilinx, Inc. All rights reserved.
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

cmake_minimum_required(VERSION 3.5)
project(morphi-unit-test VERSION 1.0.0 LANGUAGES C CXX)
include(CTest)
cmake_policy(SET CMP0135 NEW)
find_package(GTest REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_data.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/env.cmake)

configure_file(debug_logger.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/debug_logger.hpp @ONLY)
configure_file(unit_test_env_params.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/unit_test_env_params.hpp @ONLY)
## add a new test for class vaip_cxx::Model
set(TEST_EXE_NAME morphizen-unit-tests)
add_executable(${TEST_EXE_NAME}
  main.cpp
  vaip/test_config.cpp
  vaip/test_model.cpp
  vaip/test_graph.cpp
  vaip/test_const_data.cpp
  vaip/test_anchor_point.cpp
  vaip/test_immutable_map.cpp
  vaip/test_pattern.cpp
  vaip/test_pass_context.cpp
  vaip/test_node_builder.cpp
  vaip/test_tarball.cpp
  vaip/test_plugin.cpp
  vaip/test_tar_entry.cpp
  vaip/test_tar_file.cpp
  vaip/test_file_stream.cpp
  vaip/test_op_def.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/debug_logger.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/unit_test_env_params.hpp
)
set_target_properties(${TEST_EXE_NAME} PROPERTIES FOLDER morphizen/unit-tests)
find_package(onnxruntime REQUIRED)
target_link_libraries(${TEST_EXE_NAME}
    PRIVATE
    morphizen-core-static
    GTest::gtest
    onnxruntime::onnxruntime
    glog::glog
    $<LINK_LIBRARY:WHOLE_ARCHIVE,morphizen-pass-init>
)
target_include_directories(
  ${TEST_EXE_NAME}
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DONNX_NAMESPACE=onnx)
add_custom_command(
  TARGET ${TEST_EXE_NAME}
  POST_BUILD
  DEPENDS tgt_test_constant_initializer.onnx tgt_sample.onnx tgt_test_custom_op.onnx
  tgt_sample_src_tar
)
include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME} DISCOVERY_MODE "POST_BUILD" PROPERTIES
  ${TEST_ENV}
  DISCOVERY_MODE "PRE_TEST"
)
if(MSVC)
  add_custom_command(
    TARGET ${TEST_EXE_NAME}
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_vitisai.dll
      $<TARGET_FILE_DIR:${TEST_EXE_NAME}>
  )
endif(MSVC)
set_target_properties(${TEST_EXE_NAME} PROPERTIES
      VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
      VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
      VS_DEBUGGER_COMMAND_ARGUMENTS " --gtest_list_tests"
      #VS_DEBUGGER_ENVIRONMENT "${DEBUG_ENV}"
)

## for test hello_plugin test
add_library(hello_plugin_dll SHARED vaip/hello_plugin.cpp)
set_target_properties(hello_plugin_dll PROPERTIES FOLDER morphizen/unit-tests)
add_dependencies(${TEST_EXE_NAME}
  tgt_pt_resnet50.onnx
  tgt_test_constant_initializer.onnx
  tgt_sample.onnx
  tgt_sample_src_tar
  tgt_test_custom_op.onnx
  ${morphizen_CORE_DYNAMIC_UNIQUE_ID}
)
add_custom_target(morphizen-run-unit-tests
    ${CMAKE_COMMAND} -E env CI=1
    $<TARGET_FILE:${TEST_EXE_NAME}>
    DEPENDS ${TEST_EXE_NAME}
)

