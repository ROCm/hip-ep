##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-mlir-opt hip-mlir-opt.cpp)

add_dependencies(hip-mlir-opt
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

if(ENABLE_ONNX_FRONTEND)
  target_compile_definitions(hip-mlir-opt PRIVATE ENABLE_ONNX_FRONTEND)
endif()

target_link_libraries(hip-mlir-opt PRIVATE
  HipDialectIR
  UdnaHipTransforms
  UdnaConversion
  MLIROptLib
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRDestinationStyleOpInterface
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRArithTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransformUtils
  ${llvm_libs}
)

if(ENABLE_ONNX_FRONTEND)
  target_link_libraries(hip-mlir-opt PRIVATE
    UdnaConversionOnnxToHip
    -Wl,--start-group
    ${ONNX_MLIR_LIB_DIR}/libOMONNXOps.a
    ${ONNX_MLIR_LIB_DIR}/libOMMlirDialects.a
    ${ONNX_MLIR_LIB_DIR}/libOMMlirUtilities.a
    ${ONNX_MLIR_LIB_DIR}/libOMSmallFPConversion.a
    ${ONNX_MLIR_LIB_DIR}/libOMShapeHelperOpInterface.a
    ${ONNX_MLIR_LIB_DIR}/libOMONNXElementsAttr.a
    ${ONNX_MLIR_LIB_DIR}/libOMShapeInferenceOpInterface.a
    ${ONNX_MLIR_LIB_DIR}/libOMResultTypeInferenceOpInterface.a
    ${ONNX_MLIR_LIB_DIR}/libOMHasOnnxSubgraphOpInterface.a
    ${ONNX_MLIR_LIB_DIR}/libOMDiagnostic.a
    ${ONNX_MLIR_LIB_DIR}/libOMSupport.a
    ${ONNX_MLIR_LIB_DIR}/libOMAccelerator.a
    ${ONNX_MLIR_LIB_DIR}/libonnx.a
    ${ONNX_MLIR_LIB_DIR}/libonnx_proto.a
    -Wl,--end-group
    MLIRShapeDialect
    protobuf::libprotobuf
  )
endif()
