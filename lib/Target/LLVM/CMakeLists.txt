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
llvm_map_components_to_libnames(llvm_libs
  # Core infrastructure
  Support
  Core
  IRReader
  BitWriter
  Passes
  Target

  # Code generation (required by lldCOFF)
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

  # LTO and option parsing (required by lldCOFF)
  LTO
  Option

  # Windows-specific (required by lldCOFF on Windows)
  WindowsDriver
  WindowsManifest
  LibDriver
)

# Find LLD libraries using full paths
find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
find_library(LLD_ELF_LIB NAMES lldELF PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)

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
  ${LLD_COFF_LIB}
  ${LLD_ELF_LIB}
  ${LLD_COMMON_LIB}
  $<$<BOOL:${LLVM_DTLTO_LIB}>:${LLVM_DTLTO_LIB}>
)

# Set folder for IDE organization
set_target_properties(HipTargetLLVM PROPERTIES FOLDER "mlir/Target")
