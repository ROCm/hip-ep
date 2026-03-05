##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Create executable (thin wrapper around UdnaCompiler library)
add_executable(hip-compiler hip-compiler.cpp)

# Link against UdnaCompiler library (transitively links all dependencies)
# OMBuilder provides ImportFrontendModelFile for .onnx import
target_link_libraries(hip-compiler PRIVATE
  UdnaCompiler
  OMBuilder
)

# Add include directories
target_include_directories(hip-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Installation
install(TARGETS hip-compiler DESTINATION bin)
