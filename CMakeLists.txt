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

option(BUILD_EP "Build the ONNX HIP DNN Execution Provider" OFF)
option(BUILD_HIP_TOOLS "Build HIP MLIR compiler tools (hip-mlir-opt, hip-compiler)" OFF)
option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)
# Build the REAL runtime without the vendor BLAS/DNN backends (MIOpen + hipBLASLt).
# The MIOpen/hipBLASLt-backed op wrappers compile to error-returning stubs that
# preserve their ABI, and the model link + package discovery drop those libraries.
# For targets whose ROCm distribution lacks a usable MIOpen/hipBLASLt (e.g. a
# pre-silicon functional model, or a stripped SDK): the compiler still links and
# runs; only conv/gemm/hipBLASLt/MIOpen-elementwise ops are unavailable (they
# fail at runtime with a clear message unless a plugin supplies replacements).
option(HIPDNN_EP_DISABLE_VENDOR_BLAS
  "Build the real runtime without MIOpen/hipBLASLt (stub the vendor op wrappers)" OFF)
option(BUILD_HIP_UNIT_TESTS "Build and register LIT and E2E tests with CTest" ON)
option(BUILD_PYTHON_WHEEL "Configure the 'wheel' target that packages the EP as a Python wheel" OFF)

# Compiler plugins are linked STATICALLY into the host at configure time (see
# cmake/HipEpPlugins.cmake and docs/design/plugin-interface.md). Select which
# plugins to link with -DHIPDNN_EP_COMPILER_PLUGINS=<id;...> (default empty). No
# symbol export, no dynamic loader, identical on every platform. HipEpPlugins.cmake
# is included in the BUILD_HIP_TOOLS block below (it needs the LLVM/MLIR toolchain
# resolved first) and defines HIPDNN_EP_COMPILER_PLUGINS.

# Centralized dependency resolution. Always included; internally partitions by
# BUILD_EP / BUILD_HIP_TOOLS so a tools-only tree (BUILD_EP=OFF) still gets the
# shared toolchain (LLVM/MLIR/LLD, flatbuffers, cpptrace, TheRock) without the
# EP-only bits (morphizen, ORT, protobuf).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

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

if(BUILD_EP)
  # Add the GPU kernels subdirectory (lib/Runtime/Kernels) here -- driven from
  # the top-level lists, NOT from lib/Runtime/CMakeLists.txt -- so the per-arch
  # custom_kernels_<arch> SHARED targets exist before backend-mlir-compiler
  # (custom-op-mlir) declares its dependency on them. lib/Runtime is only
  # traversed later via add_subdirectory(lib) in the BUILD_HIP_TOOLS block, so
  # nesting the add there would create the targets too late for this path. The
  # guard makes a combined BUILD_EP + BUILD_HIP_TOOLS configure add it exactly
  # once (the block below is the BUILD_HIP_TOOLS-only fallback).
  if(NOT BUILD_MOCK_RUNTIME AND NOT _CUSTOM_KERNELS_SUBDIR_ADDED)
    add_subdirectory(lib/Runtime/Kernels)
    set(_CUSTOM_KERNELS_SUBDIR_ADDED TRUE)
  endif()

  # backend-mlir-compiler is added later (in the BUILD_HIP_TOOLS block, after
  # lib/Runtime so the RuntimeBitcode target exists when custom-op-mlir embeds
  # runtime.bc).
  add_subdirectory(tools/hip-onnx-runner)
endif()

# ==============================================================================
# HIP MLIR Compiler Tools (optional)
# ==============================================================================
# Build the MLIR-based HIP compiler tools (hip-mlir-opt, hip-compiler). LLVM/MLIR
# are resolved by cmake/deps.cmake (included above); enable with
# -DBUILD_HIP_TOOLS=ON.
# ==============================================================================
if(BUILD_HIP_TOOLS)
  # LLVM/MLIR, flatbuffers and cpptrace are fully resolved by cmake/deps.cmake
  # (included above for any BUILD_EP or BUILD_HIP_TOOLS tree): find_package
  # against a prefix, or a from-source FetchContent build that sets the LLVM_*/
  # MLIR_* consumer variables and helper functions directly. The block below
  # only consumes what deps.cmake already populated.
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

  # Real-runtime GPU kernels (lib/Runtime/Kernels) for a tools-only build
  # (BUILD_HIP_TOOLS without BUILD_EP). The guard skips this when the BUILD_EP
  # block above already added the subdirectory, so a combined configure wires
  # it exactly once.
  if(NOT BUILD_MOCK_RUNTIME AND NOT _CUSTOM_KERNELS_SUBDIR_ADDED)
    add_subdirectory(lib/Runtime/Kernels)
    set(_CUSTOM_KERNELS_SUBDIR_ADDED TRUE)
  endif()

  # TableGen targets
  add_subdirectory(include/hip)

  # lib/Runtime produces runtime.bc consumed by backend-mlir-compiler.
  add_subdirectory(lib)

  # Must come AFTER lib/Runtime (RuntimeBitcode dep) and BEFORE tools/
  # (tools/hip-test links morphizen-custom-op-mlir).
  if(BUILD_EP)
    add_subdirectory(backend-mlir-compiler)
  endif()

  # DLL target (hip-compiler.dll)
  add_subdirectory(dll)

  # Test-only pass registrations are linked into hip-mlir-opt when the LIT
  # suite is enabled (the default developer configuration). Production
  # packaging must configure BUILD_HIP_UNIT_TESTS=OFF to omit these probes.
  if(BUILD_HIP_UNIT_TESTS)
    add_subdirectory(test/lib/Dialect/Hip)
  endif()

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
endif()

# Python wheel packaging (opt-in; defines the 'wheel' target). Added last so
# both the EP target (morphizen_CORE_DYNAMIC_UNIQUE_ID, from deps.cmake) and the
# JIT compiler plugin target (hip-compiler-dll, from BUILD_HIP_TOOLS) exist.
if(BUILD_EP AND BUILD_PYTHON_WHEEL)
  add_subdirectory(python)
endif()
