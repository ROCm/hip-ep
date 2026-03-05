##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Passes Library")

# Create compiler passes library
add_library(UdnaCompilerPasses STATIC
  GenerateInterface.cpp
)

# Link against required libraries
target_link_libraries(UdnaCompilerPasses PUBLIC
  UdnaHipDialectIR
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
target_include_directories(UdnaCompilerPasses PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto
)

# Set folder for IDE organization
set_target_properties(UdnaCompilerPasses PROPERTIES FOLDER "mlir/Compiler")
