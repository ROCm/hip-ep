##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

# Add subdirectories
add_subdirectory(OnnxToHip)
add_subdirectory(HipToLLVM)

# Create umbrella conversion library
add_library(HipConversion STATIC
  Passes.cpp
)

# Link subdirectory libraries
target_link_libraries(HipConversion PUBLIC
  OnnxToHip
  HipToLLVM
  MLIRIR
  MLIRPass
)

# Add include directories
target_include_directories(HipConversion PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Set folder for IDE organization
set_target_properties(HipConversion PROPERTIES FOLDER "mlir/Conversion")
