##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Backend library for MLIR → LLVM IR → Object File compilation
# Supports dual modes: IR mode (text .ll) and Native mode (object file)

# Python 3 required for xxd.py (converts bitcode to C array)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Find LLD for linking
find_package(LLD REQUIRED CONFIG)

# ============================================================================
# Generate runtime_ir_data.cpp from runtime.bc
# ============================================================================

# Get runtime.bc path from Runtime project
if(NOT RUNTIME_BC_PATH)
    set(RUNTIME_BC_PATH ${CMAKE_CURRENT_BINARY_DIR}/../Runtime/runtime.bc)
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp

    # Convert runtime.bc to C array
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/cmake/scripts/xxd.py
            --var runtime_bc_data
            --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
            ${RUNTIME_BC_PATH}

    # Post-process: static const -> extern "C" const AND add size variable
    COMMAND ${Python3_EXECUTABLE} -c
            "import os; data = open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); bc_size = os.path.getsize('${RUNTIME_BC_PATH}'); data += f'\\nextern \"C\" const size_t runtime_bc_data_size = {bc_size};\\n'; open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp', 'w').write(data)"

    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp

    DEPENDS ${RUNTIME_BC_PATH}
    COMMENT "Generating runtime_ir_data.cpp from runtime.bc"
    VERBATIM
)

# Create static libraries (renamed with Morphizen prefix)
add_library(MorphizenBackend STATIC
    LLVMBackend.cpp
    DLLLinker.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp  # Generated file
)

# Include directories
target_include_directories(MorphizenBackend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}  # For generated runtime_ir_data.cpp
)

# Ensure runtime.bc is generated before building MorphizenBackend
add_dependencies(MorphizenBackend RuntimeBitcode)

# Find LLD libraries (REQUIRED - must be built with LLVM)
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
endif()

# Link LLVM and MLIR libraries
target_link_libraries(MorphizenBackend
    PUBLIC
        # MLIR translation
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRBuiltinToLLVMIRTranslation

        # LLVM core
        LLVMCore
        LLVMSupport
        LLVMTarget
        LLVMPasses

        # LLVM code generation
        LLVMCodeGen
        LLVMSelectionDAG
        LLVMAsmPrinter
        LLVMBinaryFormat
        LLVMMC
        LLVMMCParser
        LLVMTargetParser

        # Native target (will be initialized at runtime)
        LLVMX86CodeGen
        LLVMX86AsmParser
        LLVMX86Desc
        LLVMX86Info

        # LLD linker libraries (order matters - dependencies first)
        lldCommon              # Common LLD infrastructure (required by lldCOFF)
        lldCOFF                # LLD linker
        LLVMOption             # Required by lldCOFF (opt::OptTable, opt::ArgList)
        LLVMWindowsDriver      # Required by lldCOFF (Windows SDK detection)
        LLVMWindowsManifest    # Required by lldCOFF (windows_manifest::WindowsManifestMerger)
        LLVMLibDriver          # Required by lldCOFF (libDriverMain)
        LLVMLTO                # Required by lldCOFF (LTO::getRuntimeLibcallSymbols)
)

# LLVM/MLIR require these definitions
target_compile_definitions(MorphizenBackend PRIVATE
    ${LLVM_DEFINITIONS}
    HAVE_LLD=1
)

# Install rules
install(TARGETS MorphizenBackend
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES LLVMBackend.h DLLLinker.h
    DESTINATION include
)
