##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

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
  test_environment.hpp
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
  vaip/test_mmap_file.cpp
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
    Boost::process
)
target_include_directories(
  ${TEST_EXE_NAME}
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(${TEST_EXE_NAME} PRIVATE
  -DONNX_NAMESPACE=onnx
  -DPYTHON_EXE="$<TARGET_FILE:Python3::Interpreter>"
  -DTEST_CWD="${CMAKE_CURRENT_BINARY_DIR}")
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
install(
 TARGETS hello_plugin_dll
 RUNTIME DESTINATION bin
 ARCHIVE DESTINATION lib
 LIBRARY DESTINATION lib)

set_target_properties(hello_plugin_dll PROPERTIES FOLDER morphizen/unit-tests/data)
add_dependencies(${TEST_EXE_NAME}
  tgt_pt_resnet50.onnx
  tgt_test_constant_initializer.onnx
  tgt_sample.onnx
  tgt_sample_src_tar
  tgt_test_custom_op.onnx
  ${morphizen_CORE_DYNAMIC_UNIQUE_ID}
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_gtest_debug_targets.cmake)
