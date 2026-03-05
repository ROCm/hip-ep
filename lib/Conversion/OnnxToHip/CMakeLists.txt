##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building ONNX to HIP Conversion Library")

# Generate Passes.h.inc from include/hip/Conversion/OnnxToHip/Passes.td
set(TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include/hip/Conversion/OnnxToHip)
file(RELATIVE_PATH TABLEGEN_OUTPUT_REL
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${TABLEGEN_OUTPUT_DIR}"
)
file(MAKE_DIRECTORY ${TABLEGEN_OUTPUT_DIR})

set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_SOURCE_DIR}/include/hip/Conversion/OnnxToHip/Passes.td)
mlir_tablegen(${TABLEGEN_OUTPUT_REL}/Passes.h.inc -gen-pass-decls -name OnnxToHip)
add_public_tablegen_target(OnnxToHipPassesIncGen)

# Create ONNX to HIP conversion library
add_library(UdnaConversionOnnxToHip STATIC
  AddHipContextArg.cpp
  OnnxToHip.cpp
)

add_dependencies(UdnaConversionOnnxToHip OnnxToHipPassesIncGen HipDialectIncGen)

# Link against required libraries
target_link_libraries(UdnaConversionOnnxToHip PUBLIC
  HipDialectIR
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
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto  # FlatBuffers generated headers (compilation_options_generated.h)
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR source headers
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR generated headers
)

# Set folder for IDE organization
set_target_properties(UdnaConversionOnnxToHip PROPERTIES FOLDER "mlir/Conversion")
