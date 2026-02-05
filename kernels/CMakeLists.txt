##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.21)

project(rocm-kernels LANGUAGES CXX)

# Get the THEROCK_DIST path
set(THEROCK_DIST "" CACHE PATH "TheRock distribution path")

# Include HIP utilities for building .hip files
include(${CMAKE_SOURCE_DIR}/cmake/hip_utils.cmake)

# Set HIP architectures - use parent scope value if set, otherwise default to gfx1103
if(NOT HIP_ARCHITECTURES)
  set(HIP_ARCHITECTURES "gfx1103" CACHE STRING "Target GPU architectures (e.g., gfx1100, gfx1103, gfx1151)")
  message(STATUS "[rocm-kernels] HIP_ARCHITECTURES not set, defaulting to: ${HIP_ARCHITECTURES}")
else()
  message(STATUS "[rocm-kernels] Using HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")
endif()

# Create HIP kernel library using hipcc (precompiled at build time)
# Each kernel type is in its own file for better organization and maintainability
hip_add_library(rocm-hip-kernels STATIC
    src/softmax_kernels.hip      # Softmax operations (2D/3D)
    src/mul_kernels.hip          # Element-wise multiplication with broadcasting
    src/transpose_kernels.hip    # Tensor transpose (0213 optimized + general 4D)
    src/tile_kernels.hip         # N-dimensional tile/repeat operations
    src/utility_kernels.hip      # Memory copy utilities (reshape)
    src/add_bias_kernels.hip     # Bias addition for Conv (NCHW format)
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${THEROCK_DIST}/include
)

# Create C++ wrapper library (host-side code that calls HIP kernels)
add_library(rocm-kernels STATIC
    src/kernels_impl.cpp
)

# Include directories
target_include_directories(rocm-kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link HIP runtime using find_package(hip) targets
# hip::host provides headers and runtime library for host-side code
# This is the recommended CMake approach per ROCm documentation
if(TARGET hip::host)
    target_link_libraries(rocm-kernels PUBLIC hip::host)
else()
    # Fallback for older ROCm versions without hip::host target
    target_include_directories(rocm-kernels SYSTEM PRIVATE "${THEROCK_DIST}/include")
    target_compile_definitions(rocm-kernels PRIVATE "__HIP_PLATFORM_AMD__=1")
    if(WIN32)
        target_link_libraries(rocm-kernels PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
    endif()
endif()

# Link the precompiled HIP kernels
target_link_libraries(rocm-kernels PRIVATE rocm-hip-kernels)

set_target_properties(rocm-kernels PROPERTIES
    FOLDER "rocm"
    POSITION_INDEPENDENT_CODE ON
)
