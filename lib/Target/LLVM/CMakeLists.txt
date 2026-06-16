##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Target LLVM Library")

# Create target LLVM library. The embedded runtime.bc (runtime_bc_data) is
# provided by the shared runtime-bc-embed lib (lib/Runtime) so the bytes are
# generated once and shared with the EP/JIT consumer; LLVMBackend.cpp's extern
# reference pulls the embed object into the final binary.
add_library(HipTargetLLVM STATIC
  LLVMBackend.cpp
  DLLLinker.cpp
)

if(TARGET runtime-bc-embed)
    target_link_libraries(HipTargetLLVM PUBLIC runtime-bc-embed)
endif()

# Link LLVM components
#
# DLLLinker.cpp's Windows path invokes lld::lldMain in-process with the COFF
# driver. The Linux path uses a `clang++ -shared` subprocess instead (see
# linkDLL_Linux for the rationale), so it pulls in no in-process lld libs.
# The COFF driver, lldCOFF/lldCommon, and the WindowsDriver/WindowsManifest/
# LibDriver transitives are therefore Windows-only.
llvm_map_components_to_libnames(llvm_libs
  # Core infrastructure
  Support
  Core
  IRReader
  BitWriter
  Passes
  Target

  # Code generation (required by both lldCOFF and lldELF)
  CodeGen
  SelectionDAG
  AsmPrinter
  BinaryFormat
  MC
  MCParser
  TargetParser

  # X86 backend
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info

  # Linker support
  Linker

  # LTO and option parsing (required by both lldCOFF and lldELF)
  LTO
  Option
)

if(WIN32)
    # Windows COFF-only LLVM components needed by lldCOFF.
    llvm_map_components_to_libnames(llvm_libs_coff
        WindowsDriver
        WindowsManifest
        LibDriver
    )
    list(APPEND llvm_libs ${llvm_libs_coff})
endif()

# LLD: Windows only (Linux uses clang++ driver subprocess). Prefer
# find_package(LLD CONFIG) (apt.llvm.org / IMPORTED targets); fall back to
# find_library for legacy trees that omit LLDConfig.cmake.
set(lld_libs "")
if(WIN32)
    find_package(LLD CONFIG QUIET)
    if(LLD_FOUND)
        message(STATUS "Found LLD via CONFIG package: ${LLD_DIR}")
        set(lld_libs lldCommon lldCOFF)
    else()
        message(STATUS "LLD CONFIG package not found; falling back to find_library")
        find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
        find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
        list(APPEND lld_libs ${LLD_COFF_LIB} ${LLD_COMMON_LIB})
    endif()
endif()

# LLVMDTLTO was split from LLVMLTO in LLVM 22.1; link it when present.
find_library(LLVM_DTLTO_LIB NAMES LLVMDTLTO PATHS ${LLVM_LIBRARY_DIRS})

# Link against required libraries
target_link_libraries(HipTargetLLVM PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  ${llvm_libs}
  ${lld_libs}
  $<$<BOOL:${LLVM_DTLTO_LIB}>:${LLVM_DTLTO_LIB}>
)

# Bake the clang++ path used by DLLLinker.cpp's Linux subprocess link.
# DLLLinker has a PATH fallback if the baked path is stale (e.g. CI build
# deployed to a host with clang elsewhere). clang++ (not bare clang) — the
# driver auto-pulls libstdc++ which the generated model object needs for
# __cxa_begin_catch / __cxa_rethrow.
if(NOT WIN32)
    if(TARGET clang)
        # In-tree (from-source) LLVM: the clang/lld .exe are not present at
        # configure time, so find_program can't see them and would fall back to
        # a system clang (defeating the self-contained from-source build). We
        # know where the driver lands, so bake that path. clang itself is built
        # because lib/Runtime's bitcode step depends on the in-tree clang target;
        # lld is forced to build by deps.cmake (a standalone ALL target, kept out
        # of any install/export set). Mirrors lib/Runtime's in-tree-target use.
        set(HIPDNN_CLANG_EXECUTABLE "${LLVM_TOOLS_BINARY_DIR}/clang++")
        message(STATUS "Using in-tree clang++ for runtime DLL linking: ${HIPDNN_CLANG_EXECUTABLE}")
    else()
        # Prebuilt / installed LLVM (find_package path, incl. CI's llvm-install):
        # the clang++ binary already exists, so resolve it now. HINTS scope the
        # search to the LLVM tree we configure against.
        find_program(HIPDNN_CLANG_EXECUTABLE NAMES clang++
            HINTS
                ${LLVM_BINARY_DIR}/bin
                ${LLVM_TOOLS_BINARY_DIR}
        )
        if(HIPDNN_CLANG_EXECUTABLE)
            message(STATUS "Found clang++ for runtime DLL linking: ${HIPDNN_CLANG_EXECUTABLE}")
        endif()
    endif()
    if(HIPDNN_CLANG_EXECUTABLE)
        target_compile_definitions(HipTargetLLVM PRIVATE
            HIPDNN_CLANG_PATH="${HIPDNN_CLANG_EXECUTABLE}"
            HIPDNN_LLVM_MAJOR=${LLVM_VERSION_MAJOR})
    else()
        message(WARNING
            "clang not found at configure time; runtime DLL linking will "
            "rely on PATH lookup at execution time.")
    endif()
endif()

# Set folder for IDE organization
set_target_properties(HipTargetLLVM PROPERTIES FOLDER "mlir/Target")
