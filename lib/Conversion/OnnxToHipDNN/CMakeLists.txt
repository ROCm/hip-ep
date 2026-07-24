##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(ONNX_TO_HIPDNN_SOURCES OutlineOnnxToHipDNN.cpp)
set(ONNX_TO_HIPDNN_EXTRA_LIBS)

if(HIPDNN_AVAILABLE)
  list(APPEND ONNX_TO_HIPDNN_SOURCES CompileHipDNNGraphs.cpp)
  list(APPEND ONNX_TO_HIPDNN_EXTRA_LIBS HipDNNGraph)
else()
  list(APPEND ONNX_TO_HIPDNN_SOURCES CompileHipDNNGraphsStub.cpp)
endif()

add_library(OnnxToHipDNN STATIC ${ONNX_TO_HIPDNN_SOURCES})

add_dependencies(OnnxToHipDNN
  HipDialectIncGen
  HipTypesIncGen
  HipAttributesIncGen
  HipOpsIncGen
)

target_include_directories(OnnxToHipDNN PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(OnnxToHipDNN PUBLIC
  ${ONNX_TO_HIPDNN_EXTRA_LIBS}
  HipDialectIR
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRSupport
  LLVMSupport
)

set_target_properties(OnnxToHipDNN PROPERTIES FOLDER "mlir/Conversion")
