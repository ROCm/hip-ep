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
include(${CMAKE_CURRENT_SOURCE_DIR}/morphizen-e2e-test/cmake/proto.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/morphizen-e2e-test/cmake/gen-e2e-tests-config.cmake)

## add a new test for class morphizen_cxx::Model
set(TEST_EXE_NAME morphizen-unit-tests)
add_executable(${TEST_EXE_NAME}
  test_environment.hpp
  morphizen_unit_test_main.cpp
  morphizen/test_config.cpp
  morphizen/test_model.cpp
  morphizen/test_graph.cpp
  morphizen/test_const_data.cpp
  morphizen/test_anchor_point.cpp
  morphizen/test_immutable_map.cpp
  morphizen/test_pattern.cpp
  morphizen/test_pass_context.cpp
  morphizen/test_node_builder.cpp
  morphizen/test_plugin.cpp
  morphizen/test_tar_entry.cpp
  morphizen/test_tar_file.cpp
  morphizen/test_file_stream.cpp
  morphizen/test_mmap_file.cpp
  morphizen/test_mmap_tmphandle.cpp
  morphizen/test_ep_shared_context_workspace.cpp
  morphizen/test_op_invoker.cpp
  morphizen/test_mlir_ssa_dominance.cpp
  morphizen/test_onnx_return.cpp
  morphizen/test_onnx_custom.cpp
  ${MORPHIZEN_E2E_TESTS_PROTO_SRCS}
  ${MORPHIZEN_E2E_TESTS_PROTO_HDRS}
  morphizen-e2e-test/config.cpp
  morphizen-e2e-test/config.hpp
  morphizen-e2e-test/env.cpp
  morphizen-e2e-test/env.hpp
  morphizen-e2e-test/session-options.cpp
  morphizen-e2e-test/session-options.hpp
  morphizen-e2e-test/session.cpp
  morphizen-e2e-test/session.hpp
  morphizen-e2e-test/session-run.cpp
  morphizen-e2e-test/session-run.hpp
  morphizen-e2e-test/morphizen-e2e-test-main.cpp
  ${ryzenai_version_rc_file}
)

# test_onnx_schema.cpp tests ONNX schema APIs which are only available when schema support is enabled
if(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT)
  target_sources(${TEST_EXE_NAME} PRIVATE morphizen/test_onnx_schema.cpp)
endif()

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

if(morphizen_ENABLE_MLIR_BACKEND)
  target_link_libraries(${TEST_EXE_NAME} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,mlir-imp>)
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE MORPHIZEN_ENABLE_MLIR_BACKEND=1)
endif()

if(morphizen_ENABLE_BOOST)
  if(TARGET Boost::process)
    # boost 1.84 does have Boost::process, and dependencies is automatically managed.
    target_link_libraries(${TEST_EXE_NAME} PRIVATE Boost::process)
  endif()
  if(TARGET Boost::headers)
    target_link_libraries(${TEST_EXE_NAME} PRIVATE Boost::headers
      # boost 1.79.0 needs to specify the lib explicitly
      Boost::filesystem)
  endif()
  # Define compile definition for conditional compilation in source files
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE MORPHIZEN_ENABLE_BOOST)
endif()

target_include_directories(
  ${TEST_EXE_NAME}
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# ONNX headers needed for tests that use ONNX schema APIs (e.g., test_onnx_schema.cpp)
if(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT)
  target_include_directories(${TEST_EXE_NAME} PRIVATE
    $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

target_link_options(${TEST_EXE_NAME} PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>"
)

target_compile_definitions(${TEST_EXE_NAME} PRIVATE
  -DONNX_NAMESPACE=onnx
  -DPYTHON_EXE_STR="$<TARGET_FILE:Python3::Interpreter>"
  -DTEST_CWD_STR="${CMAKE_CURRENT_BINARY_DIR}"
  -DTEST_SRC_DIR_STR="${CMAKE_CURRENT_SOURCE_DIR}"
  # avoid warning message as below
  # Please define _WIN32_WINNT or _WIN32_WINDOWS appropriately. For example:
  # - add -D_WIN32_WINNT=0x0601 to the compiler command line; or
  # - add _WIN32_WINNT=0x0601 to your project's Preprocessor Definitions.
  # Assuming _WIN32_WINNT=0x0601 (i.e. Windows 7 target).
  -D_WIN32_WINNT=0x0601
  # for test onnx::OpSchema
  -DONNX_ML=1
  )

if(morphizen_ENABLE_BOOST)
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE
    -DMORPHIZEN_TAR_EXE_STR="$<TARGET_FILE:morphizen-tar>"
  )
endif()


include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME}
  DISCOVERY_TIMEOUT 600 # more stable CI test, on windows it is slow for some unknown reasons.
  DISCOVERY_MODE "PRE_TEST"
  PROPERTIES
  ${TEST_ENV}
)
if(MSVC)
  # Copy required ONNX Runtime DLLs to test directory
  # Note: onnxruntime_providers_vitisai.dll is NOT copied because this project
  # focuses on the new ABI (onnxruntime_morphizen_ep.dll built by MorphiZen).
  # The old ABI provider (onnxruntime_providers_vitisai.dll) is only available
  # when ONNX Runtime is built with --use-vitisai flag, which is not required
  # for MorphiZen development.
  add_custom_command(
    TARGET ${TEST_EXE_NAME}
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:${TEST_EXE_NAME}>
  )
endif(MSVC)
set(ENVS "")
string(APPEND ENVS "\nTEST_ONNX_MODEL=${RESNET_50_ONNX_PATH}")
set_target_properties(${TEST_EXE_NAME} PROPERTIES
      VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
      VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
      VS_DEBUGGER_COMMAND_ARGUMENTS " --gtest_list_tests"
      VS_DEBUGGER_ENVIRONMENT "${ENVS}"
)
add_dependencies(${TEST_EXE_NAME} morphizen_e2e_tests_generate_env_config)


## for test hello_plugin test
add_library(hello_plugin_dll SHARED morphizen/hello_plugin.cpp ${ryzenai_version_rc_file})
install(
 TARGETS hello_plugin_dll
 RUNTIME DESTINATION bin
 ARCHIVE DESTINATION lib
 LIBRARY DESTINATION lib)

set_target_properties(hello_plugin_dll PROPERTIES FOLDER morphizen/unit-tests/data)

# Copy test data files to build directory for tests
add_custom_command(
  TARGET ${TEST_EXE_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${RESNET_50_ONNX_PATH}
    ${CMAKE_CURRENT_BINARY_DIR}/pt_resnet50.onnx
  COMMENT "Copying pt_resnet50.onnx to test build directory"
)

if(morphizen_ENABLE_MLIR_BACKEND)
  if(EXISTS ${RESNET_50_MLIR_PATH})
    add_custom_command(
      TARGET ${TEST_EXE_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RESNET_50_MLIR_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/pt_resnet50.onnx.mlir
      COMMENT "Copying pt_resnet50.onnx.mlir to test build directory (MLIR backend)"
    )
  else()
    message(WARNING "MLIR backend enabled but MLIR test model not found: ${RESNET_50_MLIR_PATH}")
  endif()
endif()

add_dependencies(${TEST_EXE_NAME}
  tgt_pt_resnet50.onnx
  tgt_sample_src_tar
  ${morphizen_CORE_DYNAMIC_UNIQUE_ID}
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_gtest_debug_targets.cmake)
