##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP to LLVM Conversion Library")

# Create HIP to LLVM conversion library
add_library(HipToLLVM STATIC
  HipToLLVM.cpp
)

# Link against required libraries
target_link_libraries(HipToLLVM PUBLIC
  HipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  # MLIR conversion libraries
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  MLIRTransformUtils
)

# Add include directories
target_include_directories(HipToLLVM PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
)

# Set folder for IDE organization
set_target_properties(HipToLLVM PROPERTIES FOLDER "mlir/Conversion")
