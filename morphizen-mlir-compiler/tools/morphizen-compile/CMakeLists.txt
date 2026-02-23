##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Create executable (thin wrapper around MorphizenCompiler library)
add_executable(morphizen-compile morphizen-compile.cpp)

# Link against MorphizenCompiler library (transitively links all dependencies)
target_link_libraries(morphizen-compile
    PRIVATE
    MorphizenCompiler
)

# Add include directories
target_include_directories(morphizen-compile PRIVATE
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
)

# Installation
install(TARGETS morphizen-compile DESTINATION bin)
