##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(TEST_ONNX_RUNNER_PROTO_FILES
  test-onnx-runner/env_config.proto
)
set(TEST_ONNX_RUNNER_PROTO_SRCS "")
set(TEST_ONNX_RUNNER_PROTO_HDRS "")
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf CONFIG REQUIRED)
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto)
endif()
foreach(TEST_ONNX_RUNNER_PROTO_FILE ${TEST_ONNX_RUNNER_PROTO_FILES})
  get_filename_component(TEST_ONNX_RUNNER_PROTO_FILE_NAME ${TEST_ONNX_RUNNER_PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.cc ${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto/${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto/${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.h
    COMMAND protobuf::protoc
    ARGS
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/env_config_proto
    -I ${CMAKE_CURRENT_SOURCE_DIR}/test-onnx-runner
    ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_ONNX_RUNNER_PROTO_FILE}
    DEPENDS ${TEST_ONNX_RUNNER_PROTO_FILE})
  list(APPEND TEST_ONNX_RUNNER_PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto/${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.cc)
  list(APPEND TEST_ONNX_RUNNER_PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/env_config_proto/${TEST_ONNX_RUNNER_PROTO_FILE_NAME}.pb.h)
endforeach(TEST_ONNX_RUNNER_PROTO_FILE ${TEST_ONNX_RUNNER_PROTO_FILES})
