##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Pipeline Library")

# Create pipeline library (lightweight, no backend dependencies)
add_library(UdnaPipeline STATIC
  ../Pipeline.cpp
)

# Link against required libraries
# Note: Does NOT link UdnaTargetLLVM to avoid LLD dependencies
target_link_libraries(UdnaPipeline PUBLIC
  UdnaHipDialectIR
  UdnaHipTransforms
  UdnaConversion
  UdnaCompilerPasses
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRReconcileUnrealizedCasts
)

# Add include directories
target_include_directories(UdnaPipeline PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)

# Set folder for IDE organization
set_target_properties(UdnaPipeline PROPERTIES FOLDER "mlir/Compiler")
