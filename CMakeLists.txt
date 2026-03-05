##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.20)
project(udna-compiler VERSION 1.0.0 LANGUAGES CXX C)

# Build options
option(BUILD_STANDALONE_TOOLS "Build udna-opt and udna-compile" ON)
option(BUILD_MOCK_RUNTIME "Build with mock runtime (no GPU required)" ON)
option(ONNX_HIP_INCLUDE_LIT_TESTS "Include LIT tests" ON)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
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

# Build ONNX-MLIR from 3rd-party (required for OnnxToHip conversion pass)
# Only build if not already built by parent project
if(NOT TARGET OMONNXOps)
    message(STATUS "Building ONNX-MLIR from 3rd-party/onnx-mlir")
    add_subdirectory(3rd-party)

    # Fix missing dependency in onnx-mlir: OMONNXOps needs OMTransformsKrnlPassIncGen
    # This ensures PassesKrnl.h.inc is generated before OMONNXOps tries to include it
    add_dependencies(OMONNXOps OMTransformsKrnlPassIncGen)
else()
    message(STATUS "Using ONNX-MLIR from parent project")
endif()

# Mock runtime configuration
if(BUILD_MOCK_RUNTIME)
    message(STATUS "Building with MOCK runtime (no GPU required)")
    add_compile_definitions(USE_MOCK_RUNTIME)
else()
    message(STATUS "Building with REAL runtime (requires HIP/MIOpen)")
    # Find HIP, MIOpen, hipBLASLt
    if(DEFINED ENV{THEROCK_DIST})
        set(HIP_ROOT_DIR $ENV{THEROCK_DIST})
        message(STATUS "Using TheRock SDK at: ${HIP_ROOT_DIR}")
    endif()
endif()

# Proto library (must be built first)
add_subdirectory(proto)

# Static libraries (shared by all tools/DLL)
# New structure (following MLIR best practices)
add_subdirectory(lib/Dialect/Hip/IR)
add_subdirectory(lib/Dialect/Hip/Transforms)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/Target/LLVM)
add_subdirectory(lib/Compiler)

# Custom HIP kernels (GQA, RoPE, elementwise_sub) — only when building real runtime with ROCm
if(NOT BUILD_MOCK_RUNTIME)
    add_subdirectory(3rd-party/custom_kernels)
endif()

# Legacy directories (keep for now, will be removed in Phase 4)
add_subdirectory(lib/Runtime)
add_subdirectory(lib/CInterface)

# DLL (always build)
add_subdirectory(dll)

# Tools (link static libs)
if(BUILD_STANDALONE_TOOLS)
    add_subdirectory(tools)
endif()

# Tests
if(ONNX_HIP_INCLUDE_LIT_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Install headers
install(DIRECTORY include/udna-compiler
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# CMake package config (future work - commented out for now)
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/udna-compiler-config.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/udna-compiler-config.cmake
#     INSTALL_DESTINATION lib/cmake/udna-compiler
# )
# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/udna-compiler-config.cmake
#     DESTINATION lib/cmake/udna-compiler
# )
