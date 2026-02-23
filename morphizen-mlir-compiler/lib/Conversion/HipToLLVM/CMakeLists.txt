##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP to LLVM Conversion Library")

# Create HIP to LLVM conversion library
add_library(MorphizenConversionHipToLLVM STATIC
  HipToLLVM.cpp
)

# Link against required libraries
target_link_libraries(MorphizenConversionHipToLLVM PUBLIC
  MorphizenHipDialectIR
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
  # MLIR conversion libraries
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  MLIRTransformUtils
)

# Add include directories
target_include_directories(MorphizenConversionHipToLLVM PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
)

# Set folder for IDE organization
set_target_properties(MorphizenConversionHipToLLVM PROPERTIES FOLDER "mlir/Conversion")
