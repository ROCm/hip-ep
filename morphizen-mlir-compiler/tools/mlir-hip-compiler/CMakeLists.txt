##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Create executable (now just a thin wrapper around MorphizenCompiler library)
add_executable(mlir-hip-compiler main.cpp)

# Link against MorphizenCompiler library (which transitively links all dependencies)
target_link_libraries(mlir-hip-compiler
    PRIVATE
    MorphizenCompiler
)

# Installation
install(TARGETS mlir-hip-compiler DESTINATION bin)
