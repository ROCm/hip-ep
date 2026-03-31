##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHipDNN STATIC
  OutlineOnnxToHipDNN.cpp
  CompileHipDNNGraphs.cpp
)

add_dependencies(OnnxToHipDNN
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

target_include_directories(OnnxToHipDNN PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(OnnxToHipDNN PUBLIC
  HipDNNGraph
  HipDialectIR
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRSupport
  LLVMSupport
)

set_target_properties(OnnxToHipDNN PROPERTIES FOLDER "mlir/Conversion")
