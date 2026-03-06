##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(hip-compiler-lib STATIC
  HipCompiler.cpp
  InterfaceGenerator.cpp
)

add_dependencies(hip-compiler-lib
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_include_directories(hip-compiler-lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_ONNX_FRONTEND)
  target_compile_definitions(hip-compiler-lib PRIVATE ENABLE_ONNX_FRONTEND)
endif()

target_link_libraries(hip-compiler-lib PUBLIC
  HipDialectIR
  HipDialectTransforms
  HipToLLVM
  MLIROptLib
  MLIRSupport
  MLIRBytecodeReader
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
  MLIRTargetLLVMIRExport
  MLIRExecutionEngine
  MLIRTranslateLib
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMMC
  LLVMSupport
  LLVMTarget
  LLVMIRReader
  LLVMLinker
  ${llvm_libs}
)

if(ENABLE_ONNX_FRONTEND)
  target_link_libraries(hip-compiler-lib PUBLIC
    OnnxToHip
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
    MLIRShapeDialect
    protobuf::libprotobuf
  )
endif()
