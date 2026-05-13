##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Target LLVM Library")

# ============================================================================
# Runtime bitcode stub
# ============================================================================
# When RuntimeBitcode is not available (no Clang / no HIP runtime build),
# generate a stub runtime_ir_data.cpp with empty data.  LLVMBackend handles
# this gracefully (bcSize == 0 → skip linking).
if(NOT TARGET RuntimeBitcode)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
        "#include <cstddef>\n"
        "extern \"C\" const unsigned char runtime_bc_data[] = {0};\n"
        "extern \"C\" const size_t runtime_bc_data_size = 0;\n"
    )
    set(RUNTIME_IR_DATA_SRC ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)
else()
    # Python 3 required for xxd.py (converts bitcode to C array)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(NOT RUNTIME_BC_PATH)
        set(RUNTIME_BC_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../Runtime/runtime.bc)
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/cmake/xxd.py
                --var runtime_bc_data
                --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
                ${RUNTIME_BC_PATH}
        COMMAND ${Python3_EXECUTABLE} -c
                "import os; data = open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); bc_size = os.path.getsize('${RUNTIME_BC_PATH}'); data = '#include <cstddef>\\n' + data + f'\\nextern \"C\" const size_t runtime_bc_data_size = {bc_size};\\n'; open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp', 'w').write(data)"
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
        DEPENDS ${RUNTIME_BC_PATH}
        COMMENT "Generating runtime_ir_data.cpp from runtime.bc"
        VERBATIM
    )
    set(RUNTIME_IR_DATA_SRC ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)
endif()

# Create target LLVM library
add_library(HipTargetLLVM STATIC
  LLVMBackend.cpp
  DLLLinker.cpp
  ${RUNTIME_IR_DATA_SRC}
)

if(TARGET RuntimeBitcode)
    add_dependencies(HipTargetLLVM RuntimeBitcode)
endif()

# Link LLVM components
#
# DLLLinker.cpp invokes lldMain with either the COFF driver (Windows model.dll
# output) or the ELF driver (Linux model.so output), gated on _WIN32. Each
# backend pulls its own set of LLVM transitive dependencies (e.g. lldCOFF
# needs WindowsDriver/WindowsManifest/LibDriver; lldELF does not). Split the
# component list and the find_library calls along the same platform line.
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

# Locate LLD. Two shipping conventions in the wild:
#
#  (a) Source-built / install-tree layout (Windows / LLVM cross-compiled in
#      vendored 3rd-party): split static archives liblldCOFF.a, liblldELF.a,
#      liblldCommon.a co-located in LLVM_LIBRARY_DIRS. Discover with
#      find_library against bare names.
#
#  (b) Distro dev package (apt.llvm.org liblld-XX-dev on Ubuntu noble,
#      brew llvm on macOS, ...): only LLDConfig.cmake + LLDTargets.cmake
#      under lib/cmake/lld/. No bare .a files for find_library to chew on;
#      libraries are exposed exclusively as IMPORTED targets (lldCOFF,
#      lldELF, lldCommon, ...) via find_package(LLD CONFIG).
#
# Try the CONFIG path first; fall back to find_library for legacy trees
# that omit LLDConfig.cmake (older 12.x source builds, custom vendored
# minimal LLD installs).
find_package(LLD CONFIG QUIET)
if(LLD_FOUND)
    message(STATUS "Found LLD via CONFIG package: ${LLD_DIR}")
    set(lld_libs lldCommon lldELF)
    if(WIN32)
        list(APPEND lld_libs lldCOFF)
    endif()
else()
    message(STATUS "LLD CONFIG package not found; falling back to find_library")
    set(lld_libs "")
    if(WIN32)
        find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
        list(APPEND lld_libs ${LLD_COFF_LIB})
    endif()
    find_library(LLD_ELF_LIB NAMES lldELF PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
    find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
    list(APPEND lld_libs ${LLD_ELF_LIB} ${LLD_COMMON_LIB})
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

# DLLLinker.cpp's Linux path invokes `clang -shared` as a subprocess
# (driver-mediated link via ld.lld) instead of in-process lld::lldMain —
# see the long comment in linkDLL_Linux for the reason. Bake the
# discovered clang path so the runtime invocation doesn't have to rely on
# PATH (the host process running the EP may not include the llvm-XX bin
# dir). PATH lookup remains the fallback in DLLLinker.cpp if the baked
# path is stale (e.g. CI build with /usr/lib/llvm-22 deployed to a host
# where clang lives elsewhere).
if(NOT WIN32)
    # HINTS (vs PATHS) gives priority to the LLVM tree we're configuring
    # against, so a stale `/usr/bin/clang-18` won't be picked over the
    # apt llvm-22 install when both exist. Default-PATH search is still
    # the fallback if the hints don't pan out.
    # We bake the clang++ driver (not bare clang) so the runtime link
    # auto-pulls libstdc++ — the generated model object references C++
    # ABI symbols (__cxa_begin_catch, __cxa_rethrow, etc.) via the
    # libhip_custom_kernels.a archive's std::unordered_map / std::string
    # usage. With bare `clang -shared` the link errors "undefined symbol:
    # __cxa_*" on every model.
    find_program(HIPDNN_CLANG_EXECUTABLE NAMES clang++
        HINTS
            ${LLVM_BINARY_DIR}/bin
            ${LLVM_TOOLS_BINARY_DIR}
    )
    if(HIPDNN_CLANG_EXECUTABLE)
        message(STATUS "Found clang++ for runtime DLL linking: ${HIPDNN_CLANG_EXECUTABLE}")
        target_compile_definitions(HipTargetLLVM PRIVATE
            HIPDNN_CLANG_PATH="${HIPDNN_CLANG_EXECUTABLE}")
    else()
        message(WARNING
            "clang not found at configure time; runtime DLL linking will "
            "rely on PATH lookup at execution time.")
    endif()
endif()

# Set folder for IDE organization
set_target_properties(HipTargetLLVM PROPERTIES FOLDER "mlir/Target")
