##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Passes Library")

# Create compiler passes library
add_library(LibHipCompilerPasses STATIC
  GenerateInterface.cpp
)

# Link against required libraries
target_link_libraries(LibHipCompilerPasses PUBLIC
  HipDialectIR
  UdnaSchemasLib
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
  flatbuffers::flatbuffers
)

# Add include directories
target_include_directories(LibHipCompilerPasses PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto
)

# Set folder for IDE organization
set_target_properties(LibHipCompilerPasses PROPERTIES FOLDER "mlir/Compiler")
