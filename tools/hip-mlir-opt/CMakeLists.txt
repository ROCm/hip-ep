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

target_link_libraries(hip-mlir-opt PRIVATE
  HipSupport
  LibHipCompiler
  HipDialectIR
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

# Export this executable's statically-linked MLIR symbols so a plugin loaded
# via HIP_EP_PLUGINS (dlopen'd into the hip-mlir-opt process) binds to THIS
# host's single MLIR pass registry -- the exact call mlir-opt makes
# (mlir/tools/mlir-opt/CMakeLists.txt). Gated by HIPDNN_ENABLE_PLUGINS, which
# sets LLVM_ENABLE_PLUGINS so this call takes effect.
if(HIPDNN_ENABLE_PLUGINS)
  export_executable_symbols_for_plugins(hip-mlir-opt)
endif()

# Install target
install(TARGETS hip-mlir-opt
  RUNTIME DESTINATION bin
)
