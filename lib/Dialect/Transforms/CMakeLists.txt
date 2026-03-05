##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP Dialect Transforms Library")

# Create HIP dialect transforms library
add_library(UdnaHipTransforms STATIC
  MemoryPooling.cpp
  Passes.cpp
)

# Link against required libraries
target_link_libraries(UdnaHipTransforms PUBLIC
  HipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
)

# Add include directories
target_include_directories(UdnaHipTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
)

# Set folder for IDE organization
set_target_properties(UdnaHipTransforms PROPERTIES FOLDER "mlir/Dialect")
