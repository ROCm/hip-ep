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

# Create target LLVM library.
#
# After the switch to bitcode-only output, this library no longer
# performs codegen or DLL linking -- it just translates MLIR to LLVM IR,
# merges runtime.bc, optimizes, and writes binary bitcode. The COFF/ELF
# LLD drivers and the X86 / WindowsDriver / LTO LLVM components that
# DLLLinker used to need are gone from the dependency list, which trims
# ~3 MB off the linked hip-compiler.dll.
add_library(HipTargetLLVM STATIC
  LLVMBackend.cpp
  ${RUNTIME_IR_DATA_SRC}
)

if(TARGET RuntimeBitcode)
    add_dependencies(HipTargetLLVM RuntimeBitcode)
endif()

llvm_map_components_to_libnames(llvm_libs
  # Core infrastructure consumed by MLIR->LLVM-IR translation, bitcode
  # parsing/writing, and the per-module optimization pipeline.
  Support
  Core
  BitReader
  BitWriter
  IRReader
  Passes
  Linker
  TargetParser
)

target_link_libraries(HipTargetLLVM PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  ${llvm_libs}
)

# Set folder for IDE organization
set_target_properties(HipTargetLLVM PROPERTIES FOLDER "mlir/Target")
