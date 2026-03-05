##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
project(onnx-hipdnn-ep VERSION 1.0.0 LANGUAGES C CXX)
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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Build Options
# ==============================================================================
option(BUILD_MOCK_RUNTIME "Build with mock runtime (no GPU required)" ON)
option(ONNX_HIP_INCLUDE_LIT_TESTS "Include LIT tests" ON)

# ==============================================================================
# MLIR / LLVM / FlatBuffers
# ==============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(flatbuffers REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using FlatBuffers version: ${flatbuffers_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

llvm_map_components_to_libnames(llvm_libs Support Core)

# ==============================================================================
# ONNX-MLIR (from 3rd-party)
# ==============================================================================
if(NOT TARGET OMONNXOps)
    message(STATUS "Building ONNX-MLIR from 3rd-party/onnx-mlir")
    add_subdirectory(3rd-party)
    add_dependencies(OMONNXOps OMTransformsKrnlPassIncGen)
else()
    message(STATUS "Using ONNX-MLIR from parent project")
endif()

# ==============================================================================
# Mock Runtime
# ==============================================================================
if(BUILD_MOCK_RUNTIME)
    message(STATUS "Building with MOCK runtime (no GPU required)")
    add_compile_definitions(USE_MOCK_RUNTIME)
else()
    message(STATUS "Building with REAL runtime (requires HIP/MIOpen)")
    if(DEFINED ENV{THEROCK_DIST})
        set(HIP_ROOT_DIR $ENV{THEROCK_DIST})
        message(STATUS "Using TheRock SDK at: ${HIP_ROOT_DIR}")
    endif()
endif()

# ==============================================================================
# hip Compiler Libraries
# ==============================================================================
add_subdirectory(proto)
add_subdirectory(lib)

if(NOT BUILD_MOCK_RUNTIME)
    add_subdirectory(3rd-party/custom_kernels)
endif()

add_subdirectory(dll)
add_subdirectory(tools)

enable_testing()
add_subdirectory(test)

install(DIRECTORY include/hip
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

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

if(THEROCK_DIST)
  string(STRIP "${THEROCK_DIST}" THEROCK_DIST)
  message(STATUS "[onnx-hipdnn-ep] THEROCK_DIST: ${THEROCK_DIST}")

  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
endif()

# ==============================================================================
# ONNX HIP DNN Execution Provider (optional)
# ==============================================================================
option(BUILD_EP "Build the ONNX HIP DNN Execution Provider" OFF)

if(BUILD_EP)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

  add_subdirectory(kernels)

  add_subdirectory(rocm-pass-common)
  add_subdirectory(level-1-pass-rocm)

  add_subdirectory(level-2-pass-rocm-conv)
  add_subdirectory(level-2-pass-rocm-gemm)
  add_subdirectory(level-2-pass-rocm-matmul)
  add_subdirectory(level-2-pass-rocm-mul)
  add_subdirectory(level-2-pass-rocm-softmax)
  add_subdirectory(level-2-pass-rocm-reshape)
  add_subdirectory(level-2-pass-rocm-transpose)
  add_subdirectory(level-2-pass-rocm-tile)

  add_subdirectory(custom-op-rocm)
endif()

# MLIR backend (experimental, requires morphizen)
if(TARGET morphizen::morphizen-core-static)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
  add_subdirectory(backend-mlir-compiler)
endif()
