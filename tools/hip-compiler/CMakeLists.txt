##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Create executable (thin wrapper around UdnaCompiler library)
add_executable(udna-compile udna-compile.cpp)

# Link against UdnaCompiler library (transitively links all dependencies)
# OMBuilder provides ImportFrontendModelFile for .onnx import
target_link_libraries(udna-compile
    PRIVATE
    UdnaCompiler
    OMBuilder
)

# Add include directories
target_include_directories(udna-compile PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Installation
install(TARGETS udna-compile DESTINATION bin)
