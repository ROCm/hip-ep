##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-test-bc loads a per-model LLVM bitcode artifact via the BitcodeJIT
# class from morphizen-custom-op-mlir (same loader the EP uses at runtime),
# resolves the inference_* entry points, runs the model end-to-end, and
# optionally validates outputs. This is the bitcode-era replacement for
# the retired hip-test-dll harness.
add_executable(hip-test-bc
  hip-test-bc.cpp
)

# LLVM Support is still needed for JSON parsing of the embedded metadata.
# The heavy ORC stack comes in transitively via morphizen-custom-op-mlir.
llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(hip-test-bc
  PRIVATE
  morphizen-custom-op-mlir   # Provides BitcodeJIT and its LLVM ORC deps.
  HipSupport
  ${llvm_libs}
)

# BitcodeJIT resolves `hip_*` launcher symbols at JIT-link time via
# ORC's `DynamicLibrarySearchGenerator::GetForCurrentProcess`, which only
# finds them if they are exported by some module already loaded in the
# host process. The EP DLL solves this by linking `hip_custom_kernels`
# WHOLE_ARCHIVE so every launcher's `__declspec(dllexport)` directive
# fires (see backend-mlir-compiler/custom-op-mlir/CMakeLists.txt for the
# full rationale). We mirror the same trick here so hip-test-bc.exe is a
# self-contained host for the JIT'd bitcode, with no dependency on the EP
# DLL being co-located. Skipped on mock / GPU-less builds, where the
# kernels target is not registered.
if(TARGET hip_custom_kernels)
  target_link_libraries(hip-test-bc
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,hip_custom_kernels>"
  )
endif()
if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip CONFIG REQUIRED)
  target_link_libraries(hip-test-bc PRIVATE hip::host)
endif()

target_include_directories(hip-test-bc
  BEFORE PRIVATE
  ${CMAKE_SOURCE_DIR}/include   # For hip/Support/DiskFileSystem.h (must precede local/include)
  ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-foundation/include  # For morphizen-foundation/file_io.hpp
  ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src           # For BitcodeJIT.h
)

install(TARGETS hip-test-bc
  RUNTIME DESTINATION bin
)
