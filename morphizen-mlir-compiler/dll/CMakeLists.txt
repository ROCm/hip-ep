##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building morphizen-mlir-compiler.dll")

# Create the DLL target
add_library(morphizen-mlir-compiler SHARED
    dll_main.cpp
    morphizen-mlir-compiler.def
)

# Link against CInterface library (which pulls in all dependencies)
target_link_libraries(morphizen-mlir-compiler PRIVATE
    MorphizenCInterface
    # MorphizenCInterface transitively links:
    #   - MorphizenCompiler
    #   - MorphizenHipDialect (→ OMONNXOps)
    #   - MorphizenBackend
    #   - MLIR/LLVM libraries
)

# Include directories for C API headers
target_include_directories(morphizen-mlir-compiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set output name (no prefix on Windows)
# Don't override RUNTIME_OUTPUT_DIRECTORY - let it use CMAKE_RUNTIME_OUTPUT_DIRECTORY variable
set_target_properties(morphizen-mlir-compiler PROPERTIES
    OUTPUT_NAME "morphizen-mlir-compiler"
)

# Install DLL and import library
install(TARGETS morphizen-mlir-compiler
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install C API headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/morphizen-mlir-compiler
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "morphizen-mlir-compiler.dll configuration complete")
