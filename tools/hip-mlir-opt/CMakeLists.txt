##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-mlir-opt hip-mlir-opt.cpp)

add_dependencies(hip-mlir-opt
  HipDialectIncGen
  HipsrDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(hip-mlir-opt PRIVATE
  HipSupport
  LibHipCompiler
  HipDialectIR
  HipsrDialectIR
  HipDialectTransforms
  HipToLLVM
  OnnxToHip
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
  MLIRTensorInferTypeOpInterfaceImpl
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

# Install target
install(TARGETS hip-mlir-opt
  RUNTIME DESTINATION bin
)
