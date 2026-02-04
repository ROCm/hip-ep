# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.21)

# On Windows, HIP kernels need to be compiled differently
# We'll create a C++ wrapper library that can be compiled with MSVC

project(rocm-kernels LANGUAGES CXX)

# Get the THEROCK_DIST path
set(THEROCK_DIST "" CACHE PATH "TheRock distribution path")

# Create a header-only interface library for now
# The actual HIP kernel compilation will be done via hipcc externally
# or we compile C++ implementations that use HIP runtime API

add_library(rocm-kernels STATIC
    src/kernels_impl.cpp
)

# Include directories
target_include_directories(rocm-kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(WIN32)
    # On Windows, use HIP runtime via amdhip64.lib and hiprtc.lib
    target_include_directories(rocm-kernels SYSTEM PRIVATE "${THEROCK_DIST}/include")
    target_compile_definitions(rocm-kernels PRIVATE "__HIP_PLATFORM_AMD__=1")
    target_link_libraries(rocm-kernels PRIVATE 
        "${THEROCK_DIST}/lib/amdhip64.lib"
        "${THEROCK_DIST}/lib/hiprtc.lib"
    )
else()
    # On Linux, use hip::device and hiprtc
    find_package(hip REQUIRED)
    target_link_libraries(rocm-kernels PUBLIC hip::device hiprtc)
endif()

set_target_properties(rocm-kernels PROPERTIES 
    FOLDER "rocm"
    POSITION_INDEPENDENT_CODE ON
)
