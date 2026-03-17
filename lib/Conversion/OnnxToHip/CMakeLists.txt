##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building ONNX to HIP Conversion Library")

add_library(OnnxToHip STATIC
  OnnxToHip.cpp
)

add_dependencies(OnnxToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
  UdnaSchemasLib
)

target_link_libraries(OnnxToHip PUBLIC
  HipDialectIR
  UdnaSchemasLib
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)

# Add include directories (morphizen-foundation headers are vendored under include/)
target_include_directories(OnnxToHip PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto
)

# Set folder for IDE organization
set_target_properties(OnnxToHip PROPERTIES FOLDER "mlir/Conversion")
