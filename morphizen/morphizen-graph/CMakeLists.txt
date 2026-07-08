##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.16)
project(morphizen-graph
    VERSION 1.0.0
    DESCRIPTION "C++ graph wrapper utilities over MORPHIZEN_ORT_API"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Source files
set(MORPHIZEN_GRAPH_SOURCES
    src/graph.cpp
    src/node.cpp
    src/node_arg.cpp
    src/node_input.cpp
    src/node_attr.cpp
)

# Header files
set(MORPHIZEN_GRAPH_HEADERS
    include/morphizen/graph.hpp
    include/morphizen/node.hpp
    include/morphizen/node_arg.hpp
    include/morphizen/node_input.hpp
    include/morphizen/node_attr.hpp
    include/morphizen/_sanity_check.hpp
)

# Create static library
add_library(morphizen-graph STATIC
    ${MORPHIZEN_GRAPH_SOURCES}
    ${MORPHIZEN_GRAPH_HEADERS}
)

# Namespace alias
add_library(morphizen::morphizen-graph ALIAS morphizen-graph)

# Include directories
target_include_directories(morphizen-graph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(morphizen-graph
    PUBLIC
        morphizen-ort-api-ext  # MORPHIZEN_ORT_API interface
    PRIVATE
        glog::glog
)

# Compiler options
target_compile_options(morphizen-graph PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
target_compile_features(morphizen-graph PUBLIC cxx_std_17)

# Build as internal code (MORPHIZEN_EXPORT_DLL=1) even though it's a static library
# Use PRIVATE for both MORPHIZEN_EXPORT_DLL and MORPHIZEN_USE_DLL
# Consumers decide for themselves whether they're building DLL or static
target_compile_definitions(morphizen-graph
    PUBLIC
        "-DONNX_NAMESPACE=onnx"
    PRIVATE
        "-DMORPHIZEN_USE_DLL=0"
        "-DMORPHIZEN_EXPORT_DLL=1"
)

set_target_properties(morphizen-graph PROPERTIES FOLDER morphizen)
