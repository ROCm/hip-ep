##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

option(BUILD_EP "Build the ONNX HIP DNN Execution Provider" OFF)

if(BUILD_EP)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

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

  add_subdirectory(custom-op-rocm)

  # Add test directory (test_gqa can be built independently)
  add_subdirectory(test)
endif()

# ==============================================================================
# HIP MLIR Compiler Tools (optional)
# ==============================================================================
# Build the MLIR-based HIP compiler tools (hip-mlir-opt, hip-compiler).
# Requires LLVM and MLIR to be available via find_package.
#
# Usage:
#   cmake -B build -S . -DBUILD_HIP_TOOLS=ON \
#     [-DONNX_MLIR_SRC=/path/to/onnx-mlir -DONNX_MLIR_BUILD=/path/to/build]
# ==============================================================================
option(BUILD_HIP_TOOLS "Build HIP MLIR compiler tools (hip-mlir-opt, hip-compiler)" OFF)

if(BUILD_HIP_TOOLS)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

  # onnx-mlir (optional): enables --convert-onnx-to-hip pass.
  if(ONNX_MLIR_SRC)
    set(ENABLE_ONNX_FRONTEND ON)
    if(NOT ONNX_MLIR_BUILD)
      set(ONNX_MLIR_BUILD "${ONNX_MLIR_SRC}/build")
    endif()
    set(ONNX_MLIR_LIB_DIR "${ONNX_MLIR_BUILD}/Release/lib")
    message(STATUS "onnx-mlir source: ${ONNX_MLIR_SRC}")
    message(STATUS "onnx-mlir build:  ${ONNX_MLIR_BUILD}")
    include_directories(${ONNX_MLIR_SRC})
    include_directories(${ONNX_MLIR_BUILD})
    find_package(Protobuf REQUIRED)
  else()
    set(ENABLE_ONNX_FRONTEND OFF)
    message(STATUS "onnx-mlir: DISABLED (ONNX_MLIR_SRC not set)")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  llvm_map_components_to_libnames(llvm_libs Support Core)

  # TableGen targets
  add_subdirectory(include/hip)

  # Library targets
  add_subdirectory(lib)

  # Executable targets
  add_subdirectory(tools)

  # HIP compiler backend (standalone CLI + EP integration)
  add_subdirectory(backend-hip-compiler)
endif()

# MLIR backend (experimental, requires morphizen EP infrastructure)
if(BUILD_EP)
  add_subdirectory(backend-mlir-compiler)
endif()
