##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Passes Library")

# Create compiler passes library
add_library(MorphizenCompilerPasses STATIC
  GenerateInterface.cpp
  Passes.cpp
)

# Link against required libraries
target_link_libraries(MorphizenCompilerPasses PUBLIC
  MorphizenHipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
)

# Add include directories
target_include_directories(MorphizenCompilerPasses PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
)

# Set folder for IDE organization
set_target_properties(MorphizenCompilerPasses PROPERTIES FOLDER "mlir/Compiler")
