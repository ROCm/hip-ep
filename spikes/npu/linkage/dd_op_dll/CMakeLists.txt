##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0091 NEW)
project(npu_linkage_dd_op_dll LANGUAGES CXX)

# T0.2 spike (Phase 0, throwaway -- see docs/design/hybrid-npu-gpu-tasks.md).
# Standalone CMake project, deliberately built with DynamicDispatch's OWN
# toolchain settings rather than hip-ep's, per
# docs/design/hybrid-npu-gpu-design.md's "Packaging and the shim boundary":
# dynamic CRT (/MD), separate from hip-ep's static (/MT). This is the one
# target across the whole spike that must NOT inherit hip-ep's CRT default --
# that mismatch is the point being tested (see ../harness, which does use
# hip-ep's /MT default).
#
# Point CMAKE_PREFIX_PATH (or -DHYBRID_LLM_INSTALL_DIR=...) at a built
# hybrid-llm install tree, e.g.:
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
#     -DHYBRID_LLM_INSTALL_DIR=D:/path/to/hybrid-llm/install
# See ../README.md for what was actually available on this machine.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(DEFINED HYBRID_LLM_INSTALL_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${HYBRID_LLM_INSTALL_DIR}")
endif()
# nlohmann_json is a conda-env dependency of DynamicDispatch's own build
# (DynamicDispatch/env.yml), not something hybrid-llm's install tree vendors
# a CMake config for. DynamicDispatchConfig.cmake calls
# find_dependency(nlohmann_json REQUIRED) unconditionally even though
# dyn_dispatch_core itself is built SHARED (no nlohmann symbols cross the
# link boundary this target cares about), so any header-compatible config
# satisfies it. Override if your nlohmann_json lives elsewhere.
if(DEFINED NLOHMANN_JSON_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NLOHMANN_JSON_CMAKE_DIR}")
endif()

find_package(XRT CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(DynamicDispatch CONFIG REQUIRED)

add_library(dd_linkage_op SHARED
    dd_linkage_op.cpp
    dd_linkage_op.h
)
target_link_libraries(dd_linkage_op PRIVATE
    DynamicDispatch::dyn_dispatch_core
    XRT::xrt_coreutil
)
# Matches the value DynamicDispatch's own build used (see
# DynamicDispatch/build/CMakeCache.txt: DD_USE_AVX512:BOOL=ON). Header-only
# behavioral switch (ops_common/dtype_utils.h), not an ABI/layout one, but
# kept consistent to avoid any divergence from how DD itself was compiled.
target_compile_definitions(dd_linkage_op PRIVATE DD_USE_AVX512)
if(MSVC)
  target_compile_options(dd_linkage_op PRIVATE /Zc:__cplusplus /EHsc)
endif()
