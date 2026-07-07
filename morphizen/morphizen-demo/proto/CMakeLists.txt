##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(PROTO_FILES
  relu_dq.proto
)
set(PROTO_SRCS "")
set(PROTO_HDRS "")
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf CONFIG REQUIRED)
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto)
endif()
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${PROTO_FILE_NAME}.pb.cc ${PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto/${PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto/${PROTO_FILE_NAME}.pb.h
    COMMAND protobuf::protoc
    ARGS
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})


add_library (relu_dq_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(relu_dq_proto PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_proto
)
target_link_libraries(relu_dq_proto PUBLIC protobuf::libprotobuf)
set_target_properties(relu_dq_proto
	PROPERTIES
	FOLDER "relu-dq")

if(NOT MSVC)
  target_compile_options(relu_dq_proto PUBLIC "-fPIC")
else()
  # Disable C4946: Protobuf uses opaque type pattern with reinterpret_cast by design.
  # Opaque types (MessageLite <-> concrete proto types) are intentionally cast using
  # reinterpret_cast because they have the same memory layout but no inheritance relationship.
  # This is a valid use case where C4946 is a false positive.
  target_compile_options(relu_dq_proto PRIVATE /wd4946)
endif(NOT MSVC)
