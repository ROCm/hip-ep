##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building udna-compiler.dll")

# Create the DLL target
add_library(udna-compiler SHARED
    dll_main.cpp
    udna-compiler.def
)

# Link against CInterface library (which pulls in all dependencies)
target_link_libraries(udna-compiler PRIVATE
    UdnaCInterface
    # UdnaCInterface transitively links:
    #   - LibHipCompiler
    #   - UdnaHipDialect (→ OMONNXOps)
    #   - UdnaBackend
    #   - MLIR/LLVM libraries
)

# Include directories for C API headers
target_include_directories(udna-compiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set output name (no prefix on Windows)
# Don't override RUNTIME_OUTPUT_DIRECTORY - let it use CMAKE_RUNTIME_OUTPUT_DIRECTORY variable
set_target_properties(udna-compiler PROPERTIES
    OUTPUT_NAME "udna-compiler"
)

# Install DLL and import library
install(TARGETS udna-compiler
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install C API headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hip
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "udna-compiler.dll configuration complete")
