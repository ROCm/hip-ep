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

# Plugins are linked STATICALLY (see cmake/HipEpPlugins.cmake), so no symbol
# export is needed -- a statically-linked plugin shares this binary's MLIR
# instance by construction. Any plugin ids selected via HIPDNN_EP_COMPILER_-
# PLUGINS are pulled in transitively through LibHipCompiler.

# Install target. EXPORT HipDnnEpTargets emits an IMPORTED HipDnnEp::hip-mlir-opt
# target in the installed package config (install(EXPORT ...) is in the
# top-level CMakeLists after add_subdirectory(tools)).
install(TARGETS hip-mlir-opt
  EXPORT HipDnnEpTargets
  RUNTIME DESTINATION bin
)
