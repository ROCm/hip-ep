##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

# Add subdirectories
add_subdirectory(OnnxToHip)
add_subdirectory(HipToLLVM)

# Create umbrella conversion library
add_library(MorphizenConversion STATIC
  Passes.cpp
)

# Link subdirectory libraries
target_link_libraries(MorphizenConversion PUBLIC
  MorphizenConversionOnnxToHip
  MorphizenConversionHipToLLVM
  MLIRIR
  MLIRPass
)

# Add include directories
target_include_directories(MorphizenConversion PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
)

# Set folder for IDE organization
set_target_properties(MorphizenConversion PROPERTIES FOLDER "mlir/Conversion")
