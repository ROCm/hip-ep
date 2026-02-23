##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building ONNX to HIP Conversion Library")

# Create ONNX to HIP conversion library
add_library(MorphizenConversionOnnxToHip STATIC
  OnnxToHip.cpp
)

# Link against required libraries
target_link_libraries(MorphizenConversionOnnxToHip PUBLIC
  MorphizenHipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTransformUtils
  OMONNXOps  # ONNX dialect from onnx-mlir
)

# Add include directories
target_include_directories(MorphizenConversionOnnxToHip PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir   # ONNX-MLIR source headers
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir   # ONNX-MLIR generated headers
)

# Set folder for IDE organization
set_target_properties(MorphizenConversionOnnxToHip PROPERTIES FOLDER "mlir/Conversion")
