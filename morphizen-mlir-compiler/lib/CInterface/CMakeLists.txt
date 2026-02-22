##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# CInterface library - C API wrapper for CompilerPipeline
# This library provides a DLL-safe C interface for morphizen-mlir-compiler

add_library(MorphizenCInterface STATIC
    CompilerAPI.cpp
)

target_include_directories(MorphizenCInterface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Define MORPHIZEN_MLIR_COMPILER_EXPORTS for __declspec(dllexport)
target_compile_definitions(MorphizenCInterface PRIVATE
    MORPHIZEN_MLIR_COMPILER_EXPORTS
)

target_link_libraries(MorphizenCInterface PUBLIC
    MorphizenCompiler
    MorphizenProto
    protobuf::libprotobuf
    # MorphizenCompiler already pulls in:
    #   - MorphizenHipDialect (→ OMONNXOps)
    #   - MorphizenBackend
    #   - MLIR/LLVM libraries
)

# Install library
install(TARGETS MorphizenCInterface
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
