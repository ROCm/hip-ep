##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Pipeline Library")

# Create pipeline library (lightweight, no backend dependencies)
add_library(HipPipeline STATIC
  ../Pipeline.cpp
)

# Link against required libraries
target_link_libraries(HipPipeline PUBLIC
  HipDialectIR
  HipDialectTransforms
  HipConversion
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRReconcileUnrealizedCasts
)

# Add include directories
target_include_directories(HipPipeline PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
)

# Set folder for IDE organization
set_target_properties(HipPipeline PROPERTIES FOLDER "mlir/Compiler")
