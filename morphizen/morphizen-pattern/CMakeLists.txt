##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.16)
project(morphizen-pattern
    VERSION 1.0.0
    DESCRIPTION "Pattern matching library for ONNX computational graphs"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Protocol buffers
set(PROTO_FILES proto/pattern.proto)
set(PROTO_SRCS "")
set(PROTO_HDRS "")
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${PROTO_FILE_NAME}.pb.cc ${PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/morphizen
    COMMAND protobuf::protoc
    ARGS
      --proto_path=${protobuf_SOURCE_DIR}/src
      --cpp_out=dllexport_decl=MORPHIZEN_DLL_SPEC:${CMAKE_CURRENT_BINARY_DIR}/morphizen
      -I ${CMAKE_CURRENT_SOURCE_DIR}/proto
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})

if(MSVC)
  set(MORPHIZEN_DLL_SPEC "__declspec(dllexport)")
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES COMPILE_DEFINITIONS "MORPHIZEN_DLL_SPEC=${MORPHIZEN_DLL_SPEC}"
    COMPILE_FLAGS "/w")
else(MSVC)
  # Leave the dllexport decoration empty on non-MSVC toolchains. protobuf v34's
  # generated code emits `class <DLL_SPEC> [[gnu::warn_unused]] Name`; a GNU
  # __attribute__ placed before the C++11 [[...]] attribute is rejected by GCC
  # (only accepted by clang). On Linux the default symbol visibility already
  # exports these proto symbols, so an empty decoration is both correct and
  # portable across GCC and clang hosts.
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES
    COMPILE_DEFINITIONS
    "MORPHIZEN_DLL_SPEC="
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-conversion")
endif(MSVC)

# Source files
set(MORPHIZEN_PATTERN_SOURCES
    src/pattern.cpp
    src/pattern_parser.cpp
    src/pattern_node.cpp
    src/pattern_wildcard.cpp
    src/pattern_constant.cpp
    src/pattern_commutable_node.cpp
    src/pattern_or.cpp
    src/pattern_sequence.cpp
    src/pattern_where.cpp
    src/pattern_graph_input.cpp
    src/pattern_graph_output.cpp
    src/pattern_node_output_arg.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

add_library(morphizen-pattern STATIC ${MORPHIZEN_PATTERN_SOURCES})
add_library(morphizen::morphizen-pattern ALIAS morphizen-pattern)

target_include_directories(morphizen-pattern
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# Generated pattern.pb.h lives at ${CMAKE_CURRENT_BINARY_DIR}/morphizen/.
# Mark BINARY_DIR as SYSTEM PRIVATE so GCC suppresses -Wconversion when
# resolving "morphizen/pattern.pb.h" via this path. protobuf >=22's
# `Map::size() -> int` narrowing in the generated inline accessors is the
# dominant offender; without SYSTEM it tripped `-Werror -Wconversion`
# coming from MORPHIZEN_COMPILER_OPTIONS in all 7 pattern_*.cpp consumers.
# SYSTEM PRIVATE (not PUBLIC) because pattern.pb.h is an internal detail —
# no morphizen-pattern public consumer ever includes it.
target_include_directories(morphizen-pattern
    SYSTEM PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# Dependencies
target_link_libraries(morphizen-pattern
    PUBLIC
        morphizen::morphizen-graph  # Uses Graph/Node wrappers
    PRIVATE
        protobuf::libprotobuf
        glog::glog
        Microsoft.GSL::GSL
)

# Optional: Python support
if(BUILD_PYTHON)
    target_link_libraries(morphizen-pattern PRIVATE Python3::Python)
    target_compile_definitions(morphizen-pattern PRIVATE "ENABLE_PYTHON=1")
endif()

target_compile_options(morphizen-pattern PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
# Disable C4946: Protobuf uses opaque type pattern with reinterpret_cast by design.
# Opaque types (MessageLite <-> concrete proto types) are intentionally cast using
# reinterpret_cast because they have the same memory layout but no inheritance relationship.
# This is a valid use case where C4946 is a false positive.
if(MSVC)
    target_compile_options(morphizen-pattern PRIVATE /wd4946)
endif()
# Suppress C4267 (size_t → int) from protobuf-generated pattern.pb.h.
# MSVC has no "system header" concept usable here, so the per-target opt-out
# stays. The GCC/Clang equivalent is handled by the `SYSTEM PRIVATE` include
# of ${CMAKE_CURRENT_BINARY_DIR} above (pattern.pb.h resolves through that
# path; -Wconversion is automatically suppressed inside system headers).
if(MSVC)
    target_compile_options(morphizen-pattern PRIVATE /wd4267)
endif()
target_compile_features(morphizen-pattern PUBLIC cxx_std_17)

target_compile_definitions(morphizen-pattern
    PUBLIC
        "-DONNX_NAMESPACE=onnx"
    PRIVATE
        "-DMORPHIZEN_USE_DLL=0"  # Static library, no DLL export/import
        "-DMORPHIZEN_EXPORT_DLL=1")

set_target_properties(morphizen-pattern PROPERTIES FOLDER morphizen)
