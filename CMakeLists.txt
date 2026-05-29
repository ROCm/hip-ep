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

# Use /Z7 (embedded debug info) instead of /Zi (shared PDB) so that parallel
# compilations under sccache don't fight over a single .pdb file (C1041).
# Also disable incremental linking to avoid .ilk file contention (LNK1104).
if(MSVC)
  foreach(lang C CXX)
    foreach(config DEBUG RELWITHDEBINFO)
      string(REPLACE "/Zi" "/Z7" CMAKE_${lang}_FLAGS_${config} "${CMAKE_${lang}_FLAGS_${config}}")
    endforeach()
  endforeach()
  foreach(type EXE SHARED MODULE)
    string(REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO}")
  endforeach()
endif()

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
  # MLIR backend compiler
  add_subdirectory(backend-mlir-compiler)
  add_subdirectory(tools/hip-onnx-runner)
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
  find_package(flatbuffers REQUIRED CONFIG)

  # cpptrace for crash backtraces
  include(FetchContent)
  set(_saved_bsl_cpptrace ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  if(WIN32)
    set(CPPTRACE_UNWIND_WITH_WINAPI ON CACHE BOOL "" FORCE)
    set(CPPTRACE_UNWIND_WITH_DBGHELP OFF CACHE BOOL "" FORCE)
  endif()
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG v0.8.3
  )
  FetchContent_MakeAvailable(cpptrace)
  set(BUILD_SHARED_LIBS ${_saved_bsl_cpptrace})

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using FlatBuffers version: ${flatbuffers_VERSION}")

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})

  # onnx-mlir is no longer required. The convert-onnx-to-hip pass uses
  # MLIR's generic Operation API to match ONNX ops by name, so no onnx-mlir
  # headers or libraries are needed.

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  llvm_map_components_to_libnames(llvm_libs Support Core)

  # FlatBuffers schemas (GenerateInterface pass metadata)
  add_subdirectory(schemas)

  # ==============================================================================
  # Mock vs Real Runtime Configuration
  # ==============================================================================
  option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)

  if(BUILD_MOCK_RUNTIME)
    message(STATUS "Building with MOCK runtime (no GPU required)")
  else()
    message(STATUS "Building with REAL runtime (requires HIP/MIOpen)")
    # Add custom HIP kernels library for Real Runtime
    add_subdirectory(3rd-party/custom_kernels)
  endif()

  # TableGen targets
  add_subdirectory(include/hip)

  # Library targets
  add_subdirectory(lib)

  # DLL target (hip-compiler.dll)
  add_subdirectory(dll)

  # Executable targets
  add_subdirectory(tools)

  # ==============================================================================
  # HIP Unit Tests (LIT + E2E)
  # ==============================================================================
  # Controls registration of LIT (MLIR pass verification) and E2E (compile MLIR
  # to DLL and execute on GPU) tests with CTest.
  #
  # Usage:
  #   cmake -DBUILD_HIP_UNIT_TESTS=OFF ...   # disable tests
  #   ctest --test-dir <build> -C Release     # run all tests
  #   ctest -R MorphizenMLIRLitTests          # run LIT tests only
  # ==============================================================================
  option(BUILD_HIP_UNIT_TESTS "Build and register LIT and E2E tests with CTest" ON)

  if(BUILD_HIP_UNIT_TESTS)
    include(CTest)
    enable_testing()

    # LIT tests for MLIR passes
    add_subdirectory(test/lit)

    # E2E integration tests (compile MLIR → DLL → execute)
    add_subdirectory(test/e2e)

    # Memory manager unit tests (requires HIP)
    if(NOT BUILD_MOCK_RUNTIME)
      add_subdirectory(test/mm)
    endif()
  endif()
endif()
