##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP Dialect Transforms Library")

# Create HIP dialect transforms library
add_library(MorphizenHipTransforms STATIC
  MemoryPooling.cpp
  BufferDeallocation.cpp
  Passes.cpp
)

# Link against required libraries
target_link_libraries(MorphizenHipTransforms PUBLIC
  MorphizenHipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRAnalysis
)

# Add include directories
target_include_directories(MorphizenHipTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
)

# Set folder for IDE organization
set_target_properties(MorphizenHipTransforms PROPERTIES FOLDER "mlir/Dialect")
