##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.20)
# Must precede project() to affect target CRT defaults -- see the
# CMAKE_MSVC_RUNTIME_LIBRARY comment below.
cmake_policy(SET CMP0091 NEW)
project(npu_registration_spike LANGUAGES CXX)

# T0.1 spike (Phase 0, throwaway -- do not productionize; see
# docs/design/hybrid-npu-gpu-tasks.md). Standalone CMake project, deliberately
# NOT added as a subdirectory of the main hip-ep build -- spikes are not
# shipped, and this one needs SDKs (DynamicDispatch, XRT, ryzen_mm) the main
# build does not otherwise depend on.
#
# Two targets:
#
#   npu_registration_alloc_probe -- HIP only. Builds and runs on any machine
#     with ROCm/HIP, including one with no NPU (this is the target you get
#     on a machine like this one). See README.md "What ran here".
#
#   npu_registration_npu_probe -- Candidate A's NPU-touching half. Only
#     configured when DynamicDispatch and XRT resolve to real CMake import
#     targets. Do NOT stub or fake those packages to force this target to
#     "build" -- an unregistered/faked NPU leg would prove nothing about the
#     question this spike exists to answer. See README.md for how to point
#     CMake at the real SDKs on the gfx1151 NPU host.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the main hip-ep build's CRT choice (see top-level CMakeLists.txt).
# hip_utils.cmake's hipcc invocation reads this to pick /MT vs /MD for the
# .hip translation units; without it, hipcc defaults to static (/MT) while a
# plain add_executable() defaults to dynamic (/MD) and the two refuse to
# link (LNK2038 RuntimeLibrary mismatch).
if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Runtime/Kernels/cmake/hip_utils.cmake)

# gpu_touch: the one .hip translation unit either probe links against, kept
# separate so npu_registration_npu_probe -- which links MSVC-built
# DynamicDispatch/XRT/ryzen_mm static libraries -- never has to be compiled
# with hipcc's clang-cl frontend in the same TU. See gpu_touch.hpp for why;
# proving clang-cl/MSVC coexistence in one binary is T0.2's job, not T0.1's.
hip_add_library(gpu_touch STATIC
    gpu_touch.hip
    OFFLOAD_ARCHS ${HIP_ARCHITECTURES}
)
target_include_directories(gpu_touch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(npu_registration_alloc_probe alloc_probe.cpp)
target_link_libraries(npu_registration_alloc_probe PRIVATE gpu_touch hip::host)

# --- Candidate A's NPU-touching half -----------------------------------
# Package names/config-module locations below are BEST GUESSES: this
# workspace has no top-level hybrid-llm CMakeLists.txt to crib the real
# find_package incantations from (onnx_custom_ops/hybrid_llm/npu/CMakeLists.txt
# just consumes DynamicDispatch::dyn_dispatch_core / ryzen_mm::ryzen_mm /
# XRT::xrt_coreutil as pre-existing targets -- it doesn't show where they
# come from). On the NPU host, replace the three find_package() calls below
# with whatever mechanism that host's build actually uses (vcpkg toolchain
# file, an explicit <Pkg>_DIR, or find_package(... CONFIG) with
# CMAKE_PREFIX_PATH pointed at the installed SDKs) so the `if(TARGET ...)`
# guard below sees the real targets.
find_package(XRT CONFIG QUIET)
find_package(DynamicDispatch CONFIG QUIET)
find_package(ryzen_mm CONFIG QUIET)

if(TARGET XRT::xrt_coreutil AND TARGET DynamicDispatch::dyn_dispatch_core)
  add_executable(npu_registration_npu_probe npu_probe.cpp)
  target_link_libraries(npu_registration_npu_probe PRIVATE
      gpu_touch
      hip::host
      DynamicDispatch::dyn_dispatch_core
      XRT::xrt_coreutil
  )
  if(TARGET ryzen_mm::ryzen_mm)
    target_link_libraries(npu_registration_npu_probe PRIVATE ryzen_mm::ryzen_mm)
  endif()
  # Matches onnx_custom_ops/hybrid_llm/npu/CMakeLists.txt's compile
  # definitions for code that links dyn_dispatch_core.
  target_compile_definitions(npu_registration_npu_probe PRIVATE
      DD_USE_AVX512
      ORT_API_MANUAL_INIT
  )
  if(MSVC)
    target_compile_options(npu_registration_npu_probe PRIVATE "/Zc:__cplusplus")
  endif()
else()
  message(STATUS
    "npu_registration_npu_probe: DynamicDispatch and/or XRT CMake packages "
    "not found -- skipping this target. Expected on every machine except "
    "the gfx1151 NPU host with those SDKs installed; see README.md.")
endif()
