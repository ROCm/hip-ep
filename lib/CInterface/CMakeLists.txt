##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# CInterface library - C API wrapper for CompilerPipeline
# This library provides a DLL-safe C interface for hip-compiler

add_library(UdnaCInterface STATIC
    CompilerAPI.cpp
)

target_include_directories(UdnaCInterface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Define HIP_COMPILER_EXPORTS for __declspec(dllexport)
target_compile_definitions(UdnaCInterface PRIVATE
    HIP_COMPILER_EXPORTS
)

target_link_libraries(UdnaCInterface PUBLIC
    LibHipCompiler
    UdnaSchemasLib
    # LibHipCompiler already pulls in:
    #   - UdnaHipDialect (→ OMONNXOps)
    #   - UdnaBackend
    #   - MLIR/LLVM libraries
)

# Install library
install(TARGETS UdnaCInterface
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
