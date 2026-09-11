##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Standalone entry point for the `hip-ir` component: the HIP dialect and
# nothing else. No ONNX Runtime EP, no runtime, no MorphiZen, no Python, no
# ROCm -- only LLVM/MLIR. This is what makes the dialect contributable by
# teams that do not build the full EP.
#
# Three ways in:
#   1. Direct configure, to build and install the hip-ir package:
#        cmake -S cmake/hip-ir -B build/hip-ir -DMLIR_DIR=...
#        cmake --build build/hip-ir --target install
#   2. add_subdirectory() from a downstream repo wanting a source build.
#   3. FetchContent with SOURCE_SUBDIR cmake/hip-ir.

cmake_minimum_required(VERSION 3.20)

project(hip-ir VERSION 1.0.0 LANGUAGES C CXX)

# Version of the HIP-IR *contract*, deliberately independent of the
# onnx-hipdnn-ep product version: additive ops bump minor, op-semantics
# changes bump major.
set(HIP_IR_VERSION "${PROJECT_VERSION}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIP_IR_STANDALONE TRUE)
else()
  set(HIP_IR_STANDALONE FALSE)
endif()

option(HIP_IR_INSTALL "Generate install/export rules for the hip-ir package"
       ${HIP_IR_STANDALONE})

# Only impose a toolchain when we own the build. Embedders keep their own
# settings; HipDialectIR declares cxx_std_17 as a usage requirement instead.
if(HIP_IR_STANDALONE)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

get_filename_component(HIP_EP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT MLIR_FOUND)
  find_package(MLIR CONFIG REQUIRED)
endif()
if(NOT LLVM_FOUND)
  find_package(LLVM CONFIG REQUIRED)
endif()

if(NOT COMMAND mlir_tablegen)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

# HipOps.td and HipTypes.td include their siblings as "hip/Dialect/IR/*.td".
# tablegen() forwards this directory's INCLUDE_DIRECTORIES as -I, but set the
# flag explicitly too so the resolution does not depend on that behaviour.
include_directories("${HIP_EP_ROOT}/include")
list(APPEND LLVM_TABLEGEN_FLAGS "-I${HIP_EP_ROOT}/include")

add_subdirectory("${HIP_EP_ROOT}/include/hip/Dialect/IR"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/hip/Dialect/IR")
add_subdirectory("${HIP_EP_ROOT}/lib/Dialect/IR"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/IR")
