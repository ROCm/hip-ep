##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.17)

message(STATUS "Configuring MLIR Custom Op")

# Find dependencies
find_package(glog CONFIG REQUIRED)

# HIP runtime headers + amdhip64 import library are needed by
# MlirCustomOp.cpp's HIPDNN_EP_PERF instrumentation (hipEvent*, hipStream_t,
# hipDeviceSynchronize). THEROCK_DIST is on CMAKE_PREFIX_PATH, so hip::host
# resolves without reordering the parent add_subdirectory sequence.
# Mock builds have no HIP SDK; the PERF path is compiled out via
# BUILD_MOCK_RUNTIME.
if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip CONFIG REQUIRED)
endif()

# LLVM ORC JIT (BitcodeJIT.cpp). LLVM is already resolved transitively
# by `add_subdirectory(3rd-party/morphizen)` in cmake/deps.cmake; we
# re-call find_package here so `llvm_map_components_to_libnames` and the
# `LLVM_LIBRARY_DIRS` / `LLVM_INCLUDE_DIRS` variables are imported into
# this subdir's scope. The call is a no-op if LLVM was already found.
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# The set below covers everything ORC's LLJIT pulls in on Windows /
# Linux for the host target. JITLink + RuntimeDyld are both included
# because LLJIT picks RuntimeDyld by default on Windows (better COFF
# support today) but newer LLVM may switch to JITLink. Listing both
# keeps the EP DLL forward-compatible with an LLVM bump.
llvm_map_components_to_libnames(bitcode_jit_llvm_libs
  Core
  Support
  BitReader
  IRReader
  Passes
  Target
  TargetParser
  MC
  MCParser
  MCDisassembler
  ExecutionEngine
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  RuntimeDyld
  CodeGen
  SelectionDAG
  AsmPrinter
  BinaryFormat
  Object
  # Host arch -- X86 today. Add AArch64* here when we cross-build for ARM.
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
)

# Create custom op library
set(LIB_NAME morphizen-custom-op-mlir)
add_library(${LIB_NAME} STATIC
  src/plugin_registration.cpp
  src/MlirCustomOp.cpp
  src/InferenceState.cpp
  src/BitcodeJIT.cpp
)

# Define MORPHIZEN_CUSTOM_OP for conditional compilation
target_compile_definitions(${LIB_NAME} PRIVATE "MORPHIZEN_CUSTOM_OP=1")

if(BUILD_MOCK_RUNTIME)
  target_compile_definitions(${LIB_NAME} PRIVATE "BUILD_MOCK_RUNTIME=1")
endif()

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
  morphizen::morphizen-core-static
  glog::glog
  mlir_compilation_proto  # Shared protobuf metadata library
  ${bitcode_jit_llvm_libs}  # LLVM ORC stack for BitcodeJIT
)
if(NOT BUILD_MOCK_RUNTIME)
  target_link_libraries(${LIB_NAME} PRIVATE hip::host)
endif()

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/..  # For generated protobuf headers (metadata.pb.h)
  ${CMAKE_SOURCE_DIR}/include     # Shared project headers (timing.h, etc.)
  ${LLVM_INCLUDE_DIRS}            # LLVM ORC headers consumed by BitcodeJIT.cpp
)

# Link to EP target with whole archive to ensure registration
# Only link if target is defined (set by parent morphizen build system)
if(DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET AND
   TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(
    ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>"
  )

  # Custom GPU kernels live inside the signed EP DLL after the bitcode
  # switch: the JIT-loaded per-model bitcode looks up `hip_*` launcher
  # symbols (exported in onnxruntime_morphizen_ep[_with_ort_bridge].def)
  # via ORC's GetForCurrentProcess search generator, and the hipcc-emitted
  # CRT initializers in each kernel .obj register the embedded fatbin with
  # libamdhip64 on EP-DLL load. The kernels lib is added by the root
  # CMakeLists when BUILD_EP=ON and BUILD_MOCK_RUNTIME=OFF.
  #
  # Plain (non-WHOLE_ARCHIVE) link is intentional: the .def file references
  # each launcher symbol by name, which pulls in the corresponding kernel
  # .obj along with its .CRT$XCU initializer and __hip_fatbin_wrapper
  # rdata. WHOLE_ARCHIVE would still work but the morphizen-core-static
  # transitivity comment above (re multiple WHOLE_ARCHIVE markers) makes
  # the plain link the lower-risk choice.
  if(TARGET hip_custom_kernels)
    target_link_libraries(
      ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
      PRIVATE hip_custom_kernels
    )
  endif()
else()
  message(WARNING "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined or not a target - skipping custom-op-mlir registration")
endif()

set_target_properties(${LIB_NAME} PROPERTIES
  FOLDER "mlir"
)
