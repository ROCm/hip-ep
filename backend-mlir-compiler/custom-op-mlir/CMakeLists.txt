##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.17)

message(STATUS "Configuring MLIR Custom Op")

# Find dependencies
find_package(glog CONFIG REQUIRED)

# The embedded runtime.bc (runtime_bc_data) is provided by the shared
# runtime-bc-embed lib (lib/Runtime); see its rationale there. LlvmIrJit.cpp's
# extern reference pulls the embed object into the EP DLL / tools.

if(HIPDNN_EP_LINK_HIP_HOST)
  find_package(hip CONFIG REQUIRED)
endif()

# Re-import LLVM only when it comes from an external prefix. In the embedded
# (from-source FetchContent) build LLVM_DIR is unset, and a find_package here
# would wrongly resolve TheRock's LLVM from CMAKE_PREFIX_PATH -- mixing two
# LLVMs into the EP DLL and crashing LLJIT in detectHost/~TargetOptions.
# In that mode the parent scope already provides LLVM_INCLUDE_DIRS, AddLLVM,
# and the LLVM targets, so just reuse them (as lib/Compiler etc. do).
if(NOT HIPDNN_LLVM_EMBEDDED)
  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

# Both RuntimeDyld and JITLink are required: LLJIT auto-selects the object
# linking layer by triple -- RuntimeDyld on Windows (x86_64 COFF), JITLink on
# Linux (x86_64 ELF).
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
  src/LlvmIrJit.cpp
  src/LoadedArtifact.cpp
)
if(TARGET RuntimeBitcode)
  add_dependencies(${LIB_NAME} RuntimeBitcode)
endif()

# Define MORPHIZEN_CUSTOM_OP for conditional compilation
target_compile_definitions(${LIB_NAME} PRIVATE "MORPHIZEN_CUSTOM_OP=1")

if(HIPDNN_EP_LINK_HIP_HOST)
  target_compile_definitions(${LIB_NAME} PRIVATE "HIPDNN_EP_LINK_HIP_HOST=1")
endif()
if(HIPDNN_EP_LOAD_KERNEL_DLLS)
  target_compile_definitions(${LIB_NAME} PRIVATE "HIPDNN_EP_LOAD_KERNEL_DLLS=1")
endif()

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
  morphizen::morphizen-core-static
  glog::glog
  mlir_compilation_proto  # Shared protobuf metadata library
  ${bitcode_jit_llvm_libs}  # LLVM ORC stack for LlvmIrJit
  # LlvmIrJit reads the plugin registry (pluginBitcodeBuffers / pluginLibraries)
  # to add plugin bitcode + libraries to the JITDylib on the default JIT path.
  # Defined in lib/Compiler; no dependency cycle (it never links this lib).
  LibHipCompiler
)
# Native current-stream accessor: LlvmIrJit registers hipdnn_ep_get/set_current_
# stream as absolute JIT symbols, so the host must define them. The reference in
# LlvmIrJit.cpp also pulls this object into the EP DLL / hip-test / hip-inspect.
if(TARGET hipdnn-ep-tls)
  target_link_libraries(${LIB_NAME} PRIVATE hipdnn-ep-tls)
endif()
# Shared embedded runtime.bc (single source of truth, lib/Runtime). LlvmIrJit's
# extern runtime_bc_data reference pulls the embed object in.
if(TARGET runtime-bc-embed)
  target_link_libraries(${LIB_NAME} PRIVATE runtime-bc-embed)
endif()
if(HIPDNN_EP_LINK_HIP_HOST)
  target_link_libraries(${LIB_NAME} PRIVATE hip::host)
endif()

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/..  # For generated protobuf headers (metadata.pb.h)
  ${CMAKE_SOURCE_DIR}/include     # Shared project headers (timing.h, etc.)
  ${LLVM_INCLUDE_DIRS}            # LLVM ORC headers consumed by LlvmIrJit.cpp
)

# Link to EP target with whole archive to ensure registration
# Only link if target is defined (set by parent morphizen build system)
if(DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET AND
   TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(
    ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>"
  )

  # Per-arch GPU kernel libs (custom_kernels_<arch>.{dll,so}) are dlopen'd by
  # LlvmIrJit at JIT init, not linked into the EP DLL -- sequence them as
  # build deps only, so the install step co-locates them with the EP binary.
  if(HIPDNN_EP_LOAD_KERNEL_DLLS AND HIP_ARCHITECTURES)
    foreach(_arch IN LISTS HIP_ARCHITECTURES)
      if(TARGET custom_kernels_${_arch})
        add_dependencies(
          ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
          custom_kernels_${_arch}
        )
      endif()
    endforeach()

    # Linux: RPATH=$ORIGIN lets the EP .so resolve the co-located
    # libcustom_kernels_<arch>.so without LD_LIBRARY_PATH. Windows uses an
    # absolute GetModuleFileName-derived path instead, so needs no property.
    if(NOT WIN32)
      set_property(
        TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
        APPEND PROPERTY INSTALL_RPATH "$ORIGIN"
      )
      set_target_properties(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
        PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
      )
    endif()
  endif()
else()
  message(WARNING "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined or not a target - skipping custom-op-mlir registration")
endif()

set_target_properties(${LIB_NAME} PROPERTIES
  FOLDER "mlir"
)
