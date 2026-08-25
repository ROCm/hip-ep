##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0091 NEW)
project(npu_linkage_harness LANGUAGES CXX)

# T0.2 spike (Phase 0, throwaway -- see docs/design/hybrid-npu-gpu-tasks.md).
# Standalone CMake project. Uses hip-ep's OWN toolchain defaults (static CRT,
# per the top-level build) -- the counterpart to ../dd_op_dll, which
# deliberately does NOT. See ../README.md.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Only need find_package(hip)'s hip::host target (no .hip translation units
# here), so this is included purely for that side effect -- see
# ../../registration/CMakeLists.txt for the precedent.
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../lib/Runtime/Kernels/cmake/hip_utils.cmake)

if(NOT DEFINED ORT_INCLUDE_DIR OR NOT DEFINED ORT_LIB)
  message(FATAL_ERROR
    "Set -DORT_INCLUDE_DIR=<...>/prebuilt-local/include/onnxruntime and "
    "-DORT_LIB=<...>/prebuilt-local/lib/onnxruntime.lib to the same ORT "
    "hip-ep itself was built against (must match onnxruntime_morphizen_ep.dll's "
    "ORT_API_VERSION). See ../README.md.")
endif()

add_executable(npu_linkage_harness harness_main.cpp)
target_include_directories(npu_linkage_harness PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(npu_linkage_harness PRIVATE hip::host ${ORT_LIB})
