##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# protobuf::protoc / protobuf::libprotobuf are provided by cmake/deps.cmake or
# the morphizen subdirectory (find_package or from-source fallback). Only resolve
# standalone when the targets are absent.
if(NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf CONFIG REQUIRED)
endif()

get_filename_component(_proto_name metadata.proto NAME_WE)
set(PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${_proto_name}.pb.cc)
set(PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/${_proto_name}.pb.h)
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND protobuf::protoc
    ARGS
      --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/metadata.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metadata.proto)

# Create static library for proto. protobuf::libprotobuf carries its own
# include dirs as INTERFACE, so no manual Protobuf_INCLUDE_DIRS needed.
add_library(mlir_compilation_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})

# protobuf::libprotobuf propagates PROTOBUF_USE_DLLS when built shared.
# Do not set it here: protobuf checks definedness, so setting it to 0 still
# marks symbols dllimport and causes LNK2019 with static libprotobuf.lib.
target_link_libraries(mlir_compilation_proto PUBLIC protobuf::libprotobuf)
target_include_directories(mlir_compilation_proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
# Suppress C4267 (size_t→int) in protobuf-generated sources
if(MSVC)
  set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS "/wd4267")
endif()
