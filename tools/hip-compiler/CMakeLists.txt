##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# CMake target is "hip-compiler-exe" to avoid collision with the
# hip-compiler DLL target in dll/. OUTPUT_NAME keeps the binary as "hip-compiler".
add_executable(hip-compiler-exe hip-compiler.cpp)

set_target_properties(hip-compiler-exe PROPERTIES
  OUTPUT_NAME "hip-compiler"
)

# Link against LibHipCompiler library (transitively links all dependencies)
# OMBuilder provides ImportFrontendModelFile for .onnx import
target_link_libraries(hip-compiler-exe PRIVATE
  LibHipCompiler
  OMBuilder
)

# Add include directories
target_include_directories(hip-compiler-exe PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Installation
install(TARGETS hip-compiler-exe DESTINATION bin)
