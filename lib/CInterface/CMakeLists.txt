##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# CInterface library - C API wrapper for CompilerPipeline
# This library provides a DLL-safe C interface for hip-compiler

add_library(HipCInterface STATIC
    CompilerAPI.cpp
)

target_include_directories(HipCInterface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Define HIP_COMPILER_EXPORTS for __declspec(dllexport)
target_compile_definitions(HipCInterface PRIVATE
    HIP_COMPILER_EXPORTS
)

target_link_libraries(HipCInterface PUBLIC
    LibHipCompiler
    HipSchemasLib
    # LibHipCompiler already pulls in:
    #   - HipDialect (→ OMONNXOps)
    #   - HipBackend
    #   - MLIR/LLVM libraries
)

# Install library
install(TARGETS HipCInterface
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
