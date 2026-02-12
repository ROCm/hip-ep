##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# HIP Support Notes
# ==============================================================================
# CMake 3.21+ has native HIP language support, but on Windows it requires using
# Clang for ALL languages (C, C++, HIP). Since this project uses MSVC for C/C++
# and many dependencies are MSVC-compiled, we use a hybrid approach:
#
#   - Windows: Custom hipcc invocation via hip_utils.cmake
#   - Linux: CMake native HIP language support
#
# This allows us to use find_package(hip) for proper CMake targets (hip::host,
# hip::device) while maintaining MSVC compatibility on Windows.
#
# Reference: https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
# ==============================================================================

project(
  onnx-hipdnn-ep
  VERSION 1.0.0
  LANGUAGES C CXX)

# ==============================================================================
# TheRock ROCm SDK Configuration
# ==============================================================================
# THEROCK_DIST points to the TheRock ROCm SDK installation directory.
# It can be set via:
#   1. CMake cache variable: -DTHEROCK_DIST=/path/to/therock
#   2. Environment variable: THEROCK_DIST=/path/to/therock
# ==============================================================================
set(THEROCK_DIST "" CACHE PATH "TheRock ROCm SDK distribution path")
if(NOT THEROCK_DIST AND DEFINED ENV{THEROCK_DIST})
  set(THEROCK_DIST "$ENV{THEROCK_DIST}")
endif()

# Strip any trailing whitespace from THEROCK_DIST
if(THEROCK_DIST)
  string(STRIP "${THEROCK_DIST}" THEROCK_DIST)
  message(STATUS "[onnx-hipdnn-ep] THEROCK_DIST: ${THEROCK_DIST}")

  # Add to CMAKE_PREFIX_PATH for find_package
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

# ==============================================================================
# Composable Kernel (CK) Configuration (Optional)
# ==============================================================================
# CK_ROOT points to the CK source tree (header-only library).
# When available, CK Tile's naive_attention_fwd can be used as an alternative
# FMHA implementation (selectable at runtime via FMHA_USE_CK_NAIVE=1).
# Without CK, only the custom portable kernel is available (recommended for RDNA3).
#
# To obtain CK:
#   git clone https://github.com/ROCm/composable_kernel.git
# Then pass: -DCK_ROOT=/path/to/composable_kernel
# ==============================================================================
set(CK_ROOT "" CACHE PATH "Path to Composable Kernel source tree (optional, for CK Tile naive_attention_fwd)")
if(NOT CK_ROOT AND DEFINED ENV{CK_ROOT})
  set(CK_ROOT "$ENV{CK_ROOT}")
endif()
set(HAS_CK_TILE OFF)
if(CK_ROOT)
  if(EXISTS "${CK_ROOT}/include/ck_tile/core.hpp")
    set(HAS_CK_TILE ON)
    message(STATUS "[onnx-hipdnn-ep] CK_ROOT: ${CK_ROOT} (CK Tile available)")
  else()
    message(WARNING "[onnx-hipdnn-ep] CK_ROOT set but ck_tile headers not found at: ${CK_ROOT}/include/ck_tile/core.hpp\n"
      "CK Tile naive_attention_fwd will NOT be available. Using portable kernel only.")
  endif()
else()
  message(STATUS "[onnx-hipdnn-ep] CK_ROOT not set. Using portable FMHA kernel only (recommended for RDNA3).")
endif()

# HIP kernels library (compiled with hipcc on Windows, CMake HIP on Linux)
add_subdirectory(kernels)

add_subdirectory(proto)
add_subdirectory(rocm-pass-common)
add_subdirectory(level-1-pass-rocm)

# Level-2 passes for different op types
add_subdirectory(level-2-pass-rocm-conv)
add_subdirectory(level-2-pass-rocm-gemm)
add_subdirectory(level-2-pass-rocm-matmul)
add_subdirectory(level-2-pass-rocm-mul)
add_subdirectory(level-2-pass-rocm-softmax)
add_subdirectory(level-2-pass-rocm-reshape)
add_subdirectory(level-2-pass-rocm-transpose)
add_subdirectory(level-2-pass-rocm-tile)
add_subdirectory(level-2-pass-rocm-gqa)

add_subdirectory(custom-op-rocm)

# Add test directory (test_gqa can be built independently)
add_subdirectory(test)
