##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(TEST_EXE_NAME  ort-bridge-test)
find_package(GTest REQUIRED)
find_package(onnxruntime REQUIRED)
add_executable(${TEST_EXE_NAME}
  src/ort-bridge-test.cpp
  src/test-environment.hpp
  src/test-hello-ep.cpp
)
target_compile_definitions(${TEST_EXE_NAME} PRIVATE
  -DONNX_NAMESPACE=onnx
  -DPYTHON_EXE_STR="$<TARGET_FILE:Python3::Interpreter>"
  -DTEST_CWD_STR="${CMAKE_CURRENT_BINARY_DIR}"
  -DMORPHIZEN_TAR_EXE_STR="$<TARGET_FILE:morphizen-tar>"
  -DMORPHIZEN_VITISAI_EP_STR="$<TARGET_FILE:${morphizen_CORE_DYNAMIC_UNIQUE_ID}>"
  -DTEST_SRC_DIR_STR="${CMAKE_CURRENT_SOURCE_DIR}"
  # avoid warning message as below
  # Please define _WIN32_WINNT or _WIN32_WINDOWS appropriately. For example:
  # - add -D_WIN32_WINNT=0x0601 to the compiler command line; or
  # - add _WIN32_WINNT=0x0601 to your project's Preprocessor Definitions.
  # Assuming _WIN32_WINNT=0x0601 (i.e. Windows 7 target).
  -D_WIN32_WINNT=0x0601
)

target_link_libraries(${TEST_EXE_NAME}
    PRIVATE
    ort-bridge
    GTest::gtest
    onnxruntime::onnxruntime
    glog::glog
)
# add debug flags and environment variables
set(ENVS "")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_VITISAI_EP_FACTORY=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_ORT_EP_API=1")
string(APPEND ENVS "\nDEBUG_LOG_LEVEL=info")
string(APPEND ENVS "\nXLNX_ONNX_EP_VERBOSE=3")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_VITISAI_EP=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_IR_CONVERTER=5")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE=c:\\temp\\ir.onnx")

set_target_properties(${TEST_EXE_NAME} PROPERTIES
      VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
      VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
      VS_DEBUGGER_COMMAND_ARGUMENTS "--gtest_filter=HelloEpTest.CreateSession"
      VS_DEBUGGER_ENVIRONMENT "${ENVS}"
)
include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME} DISCOVERY_MODE "POST_BUILD" PROPERTIES
  ${TEST_ENV}
  DISCOVERY_MODE "PRE_TEST"
)
