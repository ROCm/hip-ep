##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Pipeline Library")

# Create pipeline library (lightweight, no backend dependencies)
add_library(MorphizenPipeline STATIC
  ../Pipeline.cpp
)

# Link against required libraries
# Note: Does NOT link MorphizenTargetLLVM to avoid LLD dependencies
target_link_libraries(MorphizenPipeline PUBLIC
  MorphizenHipDialectIR
  MorphizenHipTransforms
  MorphizenConversion
  MorphizenCompilerPasses
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRBufferizationTransforms
)

# Add include directories
target_include_directories(MorphizenPipeline PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/include
)

# Set folder for IDE organization
set_target_properties(MorphizenPipeline PROPERTIES FOLDER "mlir/Compiler")
