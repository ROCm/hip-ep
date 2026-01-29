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
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES
    COMPILE_DEFINITIONS
    "MORPHIZEN_DLL_SPEC=__attribute__((visibility(\"default\")))"
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
target_compile_features(morphizen-pattern PUBLIC cxx_std_17)

target_compile_definitions(morphizen-pattern
    PUBLIC
        "-DVAIP_USE_DLL=0"
        "-DONNX_NAMESPACE=onnx"
    PRIVATE
        "-DVAIP_EXPORT_DLL=1")

set_target_properties(morphizen-pattern PROPERTIES FOLDER morphizen)
