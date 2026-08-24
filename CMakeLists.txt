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
  VERSION 0.4.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  elseif(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
    message(FATAL_ERROR
      "CMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY} selects the dynamic "
      "MSVC runtime; released binaries must link it statically.")
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/guard:cf>)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VersionInfo.cmake)

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

# Release also gets /Z7 plus linker /DEBUG: MSVC's default Release flags carry
# no debug-info switch at all, so Release builds previously shipped with zero
# symbols. That leaves the cpptrace-based crash handler (lib/Support/CrashHandler.cpp)
# unable to resolve real function names on a crash -- every frame falls back to
# the nearest exported symbol (e.g. everything printed as "hip_get_version").
# /Z7 does not disable optimizations; it only changes what debug info the
# linker embeds into the PDB, so this has no runtime perf cost.
#
# This project uses the multi-config Visual Studio generator, where
# CMAKE_BUILD_TYPE is not read by the generated vcxprojs (the config is chosen
# at build time via --config). Editing CMAKE_CXX_FLAGS_RELEASE etc. directly
# (as done above for DEBUG/RELWITHDEBINFO) does not reliably propagate through
# the vendored MorphiZen subtree (a nested add_subdirectory with its own
# project()-style setup), so use directory-scoped add_compile_options /
# add_link_options with a $<CONFIG:Release> generator expression instead --
# same mechanism already used successfully above for /guard:cf, which does
# propagate correctly into subdirectories added after this point.
if(MSVC)
  add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:/Z7>)
  add_link_options($<$<CONFIG:Release>:/DEBUG> $<$<CONFIG:Release>:/INCREMENTAL:NO>)
endif()

option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)
# Build the REAL runtime without the vendor BLAS backend (hipBLASLt).
# The hipBLASLt-backed op wrappers compile to error-returning stubs that
# preserve their ABI, and the model link + package discovery drop that library.
# For targets whose ROCm distribution lacks a usable hipBLASLt (e.g. a
# pre-silicon functional model, or a stripped SDK): the compiler still links and
# runs; only gemm/hipBLASLt ops are unavailable (they fail at runtime with a
# clear message unless a plugin supplies replacements).
option(HIPDNN_EP_DISABLE_VENDOR_BLAS
  "Build the real runtime without hipBLASLt (stub the vendor op wrappers)" OFF)
option(BUILD_HIP_UNIT_TESTS "Build and register LIT and E2E tests with CTest" ON)
option(BUILD_PYTHON_WHEEL "Configure the 'wheel' target that packages the EP as a Python wheel" OFF)

# Compiler plugins are linked STATICALLY into the host at configure time (see
# cmake/HipEpPlugins.cmake and docs/design/plugin-interface.md). Select which
# plugins to link with -DHIPDNN_EP_COMPILER_PLUGINS=<id;...> (default empty). No
# symbol export, no dynamic loader, identical on every platform. HipEpPlugins.cmake
# is included below (after the LLVM/MLIR toolchain is resolved) and defines
# HIPDNN_EP_COMPILER_PLUGINS.

# Centralized dependency resolution (LLVM/MLIR/LLD, flatbuffers, cpptrace,
# TheRock, morphizen, ONNX Runtime, protobuf).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

# Which config source GQA sessions take by default. OFF is the offline LUT; ON
# makes them benchmark on the GPU instead (the pre-LUT behaviour). Both paths are
# always compiled in and HIPDNN_GQA_AUTOTUNE_MODE overrides per process -- this
# only moves the default, for harnesses that cannot set an environment variable.
option(HIPDNN_EP_GQA_AUTOTUNE_ONLINE_DEFAULT
       "Default GQA sessions to online GPU autotuning instead of the offline LUT" OFF)

option(HIPDNN_EP_LINK_HIP_HOST
       "Link hip::host into the EP DLL (enables HIP host APIs in EP code)" ON)
option(HIPDNN_EP_LOAD_KERNEL_DLLS
       "JIT-load per-arch custom_kernels_<arch>.{dll,so} at session create" ON)

if(BUILD_MOCK_RUNTIME)
  set(HIPDNN_EP_LINK_HIP_HOST OFF)
  set(HIPDNN_EP_LOAD_KERNEL_DLLS OFF)
endif()

if(HIPDNN_EP_LOAD_KERNEL_DLLS AND NOT HIPDNN_EP_LINK_HIP_HOST)
  message(FATAL_ERROR
    "HIPDNN_EP_LOAD_KERNEL_DLLS=ON requires HIPDNN_EP_LINK_HIP_HOST=ON "
    "(LlvmIrJit::detectCustomKernelArch needs hipGetDeviceProperties).")
endif()

# Add the GPU kernels subdirectory (lib/Runtime/Kernels) here -- driven from
# the top-level lists, NOT from lib/Runtime/CMakeLists.txt -- so the per-arch
# custom_kernels_<arch> SHARED targets exist before backend-mlir-compiler
# (custom-op-mlir) declares its dependency on them. lib/Runtime is only
# traversed later via add_subdirectory(lib), so nesting the add there would
# create the targets too late.
if(NOT BUILD_MOCK_RUNTIME AND NOT _CUSTOM_KERNELS_SUBDIR_ADDED)
  add_subdirectory(lib/Runtime/Kernels)
  set(_CUSTOM_KERNELS_SUBDIR_ADDED TRUE)
endif()

add_subdirectory(tools/hip-onnx-runner)

# ==============================================================================
# HIP MLIR compiler + EP
# ==============================================================================
# LLVM/MLIR, flatbuffers and cpptrace are fully resolved by cmake/deps.cmake
# (included above): find_package against a prefix, or a from-source FetchContent
# build that sets the LLVM_*/MLIR_* consumer variables and helper functions
# directly. The block below only consumes what deps.cmake already populated.
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

# The convert-onnx-to-hip pass matches ONNX ops by name via MLIR's generic
# Operation API, so no onnx-mlir headers or libraries are needed.

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Static compiler-plugin machinery: HIPDNN_EP_COMPILER_PLUGINS cache var +
# hipdnn_ep_compiler_plugin_register() + hipdnn_ep_finalize_static_plugins().
# Included here (after the LLVM/MLIR toolchain is resolved) and before
# add_subdirectory(lib)/(tools)/(test) so plugin packages can register.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipEpPlugins.cmake)

llvm_map_components_to_libnames(llvm_libs Support Core)

# FlatBuffers schemas (GenerateInterface pass metadata)
add_subdirectory(schemas)

# TableGen targets
add_subdirectory(include/hip)

# lib/Runtime produces runtime.bc consumed by backend-mlir-compiler.
add_subdirectory(lib)

# Must come AFTER lib/Runtime (RuntimeBitcode dep) and BEFORE tools/
# (tools/hip-test links morphizen-custom-op-mlir).
add_subdirectory(backend-mlir-compiler)

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
if(BUILD_HIP_UNIT_TESTS)
  include(CTest)
  enable_testing()

  # LIT tests for MLIR passes
  add_subdirectory(test/lit)

  # E2E integration tests (compile MLIR → DLL → execute)
  add_subdirectory(test/e2e)

  # Runtime unit tests (GPU-free; native against mock types)
  add_subdirectory(test/runtime)

  # Static compiler-plugin tests (sample plugin + unit test).
  # See docs/design/plugin-interface.md.
  add_subdirectory(test/plugin)

  # MLIR dialect/pass unit tests (plain main; GPU-free injected-I/O coverage).
  add_subdirectory(test/unit)
endif()

# Out-of-tree plugin dirs (HIPDNN_EP_COMPILER_PLUGIN_PATHS): add each so it can
# register its plugin. Must run before finalize (below).
hipdnn_ep_add_external_plugins()

# Finalize static compiler plugins: generate the registrar include and link
# the selected plugin static libs into LibHipCompiler. Must run after every
# plugin package (in-tree: test/plugin; out-of-tree: the paths above) has
# registered.
hipdnn_ep_finalize_static_plugins()

# Hide the statically-absorbed LLVM/MLIR symbols from each host executable's
# dynamic table so they cannot interpose ROCm comgr's private libLLVM at HIP
# init (see hipdnn_ep_harden_comgr_safe() in cmake/HipEpPlugins.cmake). Applied
# unconditionally, not just when a plugin is selected: the tools link LLVM/MLIR
# statically regardless, so the interposition hazard is latent in every build.
foreach(_host_exe hip-test hip-compiler hip-mlir-opt hip-onnx-runner hip-inspect)
  if(TARGET ${_host_exe})
    hipdnn_ep_harden_comgr_safe(${_host_exe})
  endif()
endforeach()

# Public C API headers (compiler_api.h, compiler_types.h, ...).
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hip
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Python wheel packaging (opt-in; defines the 'wheel' target). Added last so the
# EP target (morphizen_CORE_DYNAMIC_UNIQUE_ID, from deps.cmake) exists. The JIT
# compiler is linked into that target, so there is no separate compiler library
# to stage.
if(BUILD_PYTHON_WHEEL)
  add_subdirectory(python)
endif()
