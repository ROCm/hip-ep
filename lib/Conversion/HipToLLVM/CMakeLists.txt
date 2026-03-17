##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP to LLVM Conversion Library")

# Create HIP to LLVM conversion library
add_library(UdnaConversionHipToLLVM STATIC
  HipToLLVM.cpp
)

# Link against required libraries
target_link_libraries(UdnaConversionHipToLLVM PUBLIC
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
target_include_directories(UdnaConversionHipToLLVM PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
)

# Set folder for IDE organization
set_target_properties(UdnaConversionHipToLLVM PROPERTIES FOLDER "mlir/Conversion")
