# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# ROCm HIP Kernels Library
# ========================
# This library provides precompiled HIP kernels for GPU-accelerated operations:
#   - Softmax (2D/3D with double-precision accumulation)
#   - Element-wise Mul (with broadcasting)
#   - Transpose (optimized 0213 and general 4D)
#   - Tile (N-dimensional)
#   - Reshape (memory copy)
#   - Add Bias (NCHW format for Conv)
#
# Build Strategy:
# ---------------
# - Windows: Uses hip_utils.cmake with custom hipcc invocation (MSVC + hipcc)
# - Linux:   Uses CMake native HIP language support (enable_language(HIP))
#
# This hybrid approach allows MSVC compatibility on Windows while still using
# find_package(hip) for proper CMake targets (hip::host).
#
# Reference: https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
#
# IMPORTANT: GPU Architecture Selection
# -------------------------------------
# HIP_ARCHITECTURES must match your target GPU. Using the wrong architecture
# will result in runtime crashes (access violation in hipLaunchKernel).
#
# Common AMD GPU architectures:
#   - gfx1100: Radeon RX 7900 series (Navi 31)
#   - gfx1103: Radeon 780M/760M integrated graphics (Phoenix)
#   - gfx1151: Radeon PRO W7000 series
#   - gfx90a:  MI200 series (CDNA2)
#   - gfx942:  MI300 series (CDNA3)
#
# Set architecture from command line:
#   cmake -DHIP_ARCHITECTURES=gfx1151 <build_dir>
#
# Or use CMAKE_HIP_ARCHITECTURES on Linux with native HIP:
#   cmake -DCMAKE_HIP_ARCHITECTURES=gfx1151 <build_dir>

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
