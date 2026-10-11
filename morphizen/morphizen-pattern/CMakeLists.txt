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

# Dependencies
target_link_libraries(morphizen-pattern
    PUBLIC
        morphizen::morphizen-graph  # Uses Graph/Node wrappers
    PRIVATE
        morphizen-utils
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
        "-DONNX_NAMESPACE=onnx"
    PRIVATE
        "-DMORPHIZEN_USE_DLL=0"  # Static library, no DLL export/import
        "-DMORPHIZEN_EXPORT_DLL=1")

set_target_properties(morphizen-pattern PROPERTIES FOLDER morphizen)
