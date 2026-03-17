##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

# Add subdirectories
add_subdirectory(OnnxToHip)
add_subdirectory(HipToLLVM)

# Create umbrella conversion library
add_library(UdnaConversion STATIC
  Passes.cpp
)

# Link subdirectory libraries
target_link_libraries(UdnaConversion PUBLIC
  OnnxToHip
  UdnaConversionHipToLLVM
  MLIRIR
  MLIRPass
)

# Add include directories
target_include_directories(UdnaConversion PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Set folder for IDE organization
set_target_properties(UdnaConversion PROPERTIES FOLDER "mlir/Conversion")
