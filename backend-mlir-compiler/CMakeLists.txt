##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)

message(STATUS "Configuring MLIR Compiler Backend")

# Protobuf library for metadata serialization
find_package(Protobuf REQUIRED)

# Generate protobuf sources in build directory
set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/proto/mlir_compilation.proto)
protobuf_generate_cpp(MLIR_COMPILATION_PROTO_SRC MLIR_COMPILATION_PROTO_HDR ${PROTO_FILES})

# Create protobuf library
add_library(mlir_compilation_proto STATIC ${MLIR_COMPILATION_PROTO_SRC} ${MLIR_COMPILATION_PROTO_HDR})
if(MSVC)
  # CRITICAL: Link debug protobuf in Debug mode, release in Release mode
  target_link_libraries(mlir_compilation_proto PUBLIC
    $<$<CONFIG:Debug>:${CMAKE_PREFIX_PATH}/lib/libprotobufd.lib>
    $<$<CONFIG:Release>:${CMAKE_PREFIX_PATH}/lib/libprotobuf.lib>
    $<$<CONFIG:MinSizeRel>:${CMAKE_PREFIX_PATH}/lib/libprotobufd.lib>
    $<$<CONFIG:RelWithDebInfo>:${CMAKE_PREFIX_PATH}/lib/libprotobufd.lib>
  )
else()
  target_link_libraries(mlir_compilation_proto PUBLIC protobuf::libprotobuf)
endif()
target_include_directories(mlir_compilation_proto PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}  # Generated .pb.h files are here
  ${Protobuf_INCLUDE_DIRS}
)
# Fix LNK4217 warnings: protobuf is statically linked, not a DLL
target_compile_definitions(mlir_compilation_proto PRIVATE PROTOBUF_USE_DLLS=0)

# Level-1 Pass (MorphiZen integration - always build)
add_subdirectory(level-1-pass)

# Custom Op MLIR (MorphiZen runtime - always build)
add_subdirectory(custom-op-mlir)
