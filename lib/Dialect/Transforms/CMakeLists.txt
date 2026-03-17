##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectTransforms STATIC
  AddHipContextArg.cpp
  BufferUtils.cpp
  LowerAllocs.cpp
  OptimizeMemRefs.cpp
  Pipelines.cpp
  PoolAllocs.cpp
  ResolveExternConstants.cpp
)

add_dependencies(HipDialectTransforms
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  HipToLLVM
  OnnxToHip
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
)
