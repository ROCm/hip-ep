##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Third-party dependencies for morphizen-mlir-compiler

# CRITICAL: Disable ALL tests in onnx-mlir and its dependencies
# This prevents CMake from looking for LLVM test utilities (FileCheck, not, count, etc.)

# Disable ONNX-MLIR tests
set(ONNX_MLIR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_MLIR_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(ONNX_MLIR_BUILD_DOC OFF CACHE BOOL "" FORCE)

# Disable LLVM/MLIR tests (inherited by onnx-mlir)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLIR_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Disable StableHLO tests (onnx-mlir dependency)
set(STABLEHLO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_MLIR_ENABLE_STABLEHLO OFF CACHE BOOL "" FORCE)

# Disable benchmark builds
set(ONNX_MLIR_ENABLE_BENCHMARK OFF CACHE BOOL "" FORCE)

# Suppress CMake deprecation warnings from third-party onnx-mlir
#
# PROBLEM: onnx-mlir sets cmake_policy(SET CMP0116 OLD) at line 37 of its CMakeLists.txt
# to maintain compatibility with LLVM's build system. CMP0116 controls how Ninja generators
# handle DEPFILE outputs from add_custom_command() (used by LLVM TableGen).
#
# WHY SUPPRESS: CMake 3.28+ warns that OLD behavior will be removed in future versions.
# This warning appears during every cmake configuration and cannot be fixed without modifying
# third-party code.
#
# HOW IT WORKS: CMAKE_WARN_DEPRECATED=FALSE is inherited by the onnx-mlir subdirectory scope
# but does NOT affect the parent project or other subdirectories. We explicitly restore it
# to TRUE immediately after to prevent any leakage.
#
# WHEN TO REMOVE: Remove this suppression when onnx-mlir upstream either:
#   1. Updates to cmake_policy(SET CMP0116 NEW), or
#   2. Removes the policy setting entirely, or
#   3. Bumps cmake_minimum_required() to 3.20+ (which auto-defaults CMP0116 to NEW)
#

# Suppress CMake deprecation warnings from third-party onnx-mlir
set(CMAKE_WARN_DEPRECATED FALSE)

# Add ONNX-MLIR (required for ONNX dialect and OnnxToHip conversion)
# onnx-mlir submodule is now local to morphizen-mlir-compiler/3rd-party
# Note: The test directory issue is now fixed directly in onnx-mlir/CMakeLists.txt
# (see ONNX_MLIR_FORK_SUMMARY.md Change 6)
add_subdirectory(onnx-mlir EXCLUDE_FROM_ALL)

# Restore deprecation warnings
set(CMAKE_WARN_DEPRECATED TRUE)

# Fix missing dependencies in onnx-mlir (minimal modification)
# OMONNXOps needs PassesKrnl.h.inc to be generated first
if(TARGET OMONNXOps AND TARGET OMTransformsKrnlPassIncGen)
  add_dependencies(OMONNXOps OMTransformsKrnlPassIncGen)
endif()
