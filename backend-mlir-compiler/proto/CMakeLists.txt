##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(Protobuf REQUIRED)
# Generate C++ code from proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metadata.proto)

# Create static library for proto
add_library(mlir_compilation_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})

target_link_libraries(mlir_compilation_proto PUBLIC protobuf::libprotobuf)
target_include_directories(mlir_compilation_proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)
# Suppress C4267 (size_t→int) in protobuf-generated sources
if(MSVC)
  set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS "/wd4267")
endif()
# Fix LNK4217 warnings: protobuf is statically linked, not a DLL
target_compile_definitions(mlir_compilation_proto PRIVATE PROTOBUF_USE_DLLS=0)
