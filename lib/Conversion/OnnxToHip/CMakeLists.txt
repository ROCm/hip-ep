##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building ONNX to HIP Conversion Library")

# Create ONNX to HIP conversion library
add_library(UdnaConversionOnnxToHip STATIC
  AddHipContextArg.cpp
  OnnxToHip.cpp
)

# Link against required libraries
target_link_libraries(UdnaConversionOnnxToHip PUBLIC
  UdnaHipDialectIR
  UdnaSchemasLib  # ensures compilation_options_generated.h is generated first
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRTransformUtils
  OMONNXOps  # ONNX dialect from onnx-mlir
)

# Add include directories
target_include_directories(UdnaConversionOnnxToHip PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto  # FlatBuffers generated headers (compilation_options_generated.h)
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR source headers
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR generated headers
)

# Set folder for IDE organization
set_target_properties(UdnaConversionOnnxToHip PROPERTIES FOLDER "mlir/Conversion")
