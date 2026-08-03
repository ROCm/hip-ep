##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(GTest REQUIRED)
find_package(onnxruntime REQUIRED)
set(TEST_EXE_NAME ort-bridge-test)
add_executable(${TEST_EXE_NAME}
  src/ort-bridge-test.cpp
  src/test-environment.hpp
  src/test-hello-ep.cpp
  src/test-compile-model.cpp
  src/test-coverage-wrapper.cpp
  src/test-morphizen-ort-implementation.cpp
  # test-ort-array-span.cpp
  # test-graph-classes.cpp
  src/test-coverage-wrapper.hpp
  src/test-coverage-wrapper.cpp
  src/test-api-coverage-checker.hpp
)

target_include_directories(${TEST_EXE_NAME}
    PRIVATE
    $<BUILD_INTERFACE:${MORPHIZEN_ORT_API_DIR}>
    $<INSTALL_INTERFACE:../include>
)
target_compile_definitions(${TEST_EXE_NAME} PRIVATE
  -DPYTHON_EXE_STR="$<TARGET_FILE:Python3::Interpreter>"
  -DTEST_CWD_STR="${CMAKE_CURRENT_BINARY_DIR}"
  -DMORPHIZEN_MORPHIZEN_EP_STR="$<TARGET_FILE:${morphizen_CORE_DYNAMIC_UNIQUE_ID}>"
  -DTEST_SRC_DIR_STR="${CMAKE_CURRENT_SOURCE_DIR}"
  # avoid warning message as below
  # Please define _WIN32_WINNT or _WIN32_WINDOWS appropriately. For example:
  # - add -D_WIN32_WINNT=0x0601 to the compiler command line; or
  # - add _WIN32_WINNT=0x0601 to your project's Preprocessor Definitions.
  # Assuming _WIN32_WINNT=0x0601 (i.e. Windows 7 target).
  -D_WIN32_WINNT=0x0601
  # from test point of view, ort-bridge can only see public onnx definitions in my_ort.h
)
if(morphizen_ENABLE_BOOST)
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE
    MORPHIZEN_TAR_EXE_STR="$<TARGET_FILE:morphizen-tar>"
  )
endif()

target_link_options(${TEST_EXE_NAME} PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>"
)

target_link_libraries(${TEST_EXE_NAME}
    PRIVATE
    GTest::gtest
    onnxruntime::onnxruntime
    glog::glog
    morphizen-core-static
    morphizen-ort-api-ext
    $<GENEX_EVAL:$<TARGET_PROPERTY:${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET},LINK_LIBRARIES>>
)
# add debug flags and environment variables
set(ENVS "")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_MORPHIZEN_EP_FACTORY=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_ORT_EP_API=1")
string(APPEND ENVS "\nDEBUG_LOG_LEVEL=info")
string(APPEND ENVS "\nHIP_EP_VERBOSE=3")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_MORPHIZEN_EP=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_IR_CONVERTER=5")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE=c:\\temp\\ir.onnx")

set_target_properties(${TEST_EXE_NAME} PROPERTIES
      VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
      VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
      VS_DEBUGGER_COMMAND_ARGUMENTS "--gtest_filter=HelloEpTest.CreateSession"
      VS_DEBUGGER_ENVIRONMENT "${ENVS}"
      WIN32_EXECUTABLE FALSE  # Ensures console subsystem
)
include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME} DISCOVERY_MODE "POST_BUILD"
  DISCOVERY_TIMEOUT 600
  DISCOVERY_MODE "PRE_TEST"
  PROPERTIES
  ${TEST_ENV}
)
