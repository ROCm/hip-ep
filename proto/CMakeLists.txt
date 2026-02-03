##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(PROTO_FILES
  hipdnn.proto
)
set(PROTO_SRCS "")
set(PROTO_HDRS "")
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf CONFIG REQUIRED)
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto)
endif()
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${PROTO_FILE_NAME}.pb.cc ${PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto/${PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto/${PROTO_FILE_NAME}.pb.h
    COMMAND protobuf::protoc
    ARGS
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})


add_library (hipdnn_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(hipdnn_proto PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_proto
)
target_link_libraries(hipdnn_proto PUBLIC protobuf::libprotobuf)
set_target_properties(hipdnn_proto
	PROPERTIES
	FOLDER "hipdnn")

if(NOT MSVC)
  target_compile_options(hipdnn_proto PUBLIC "-fPIC")
endif(NOT MSVC)
