##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectTransforms STATIC
  AddHipContextArg.cpp
  BufferUtils.cpp
  ComposeDimSpecs.cpp
  GenerateInterface.cpp
  LowerAllocs.cpp
  OptimizeMemRefs.cpp
  Pipelines.cpp
  PoolAllocs.cpp
  PromoteStridedHipOperands.cpp
  ResolveExternConstants.cpp
)

add_dependencies(HipDialectTransforms
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
  HipSchemas
)

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  HipToLLVM
  OnnxToHip
  OnnxToHipDNN
  HipSchemasLib
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  flatbuffers::flatbuffers
)
