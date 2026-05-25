##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectTransforms STATIC
  AddHipContextArg.cpp
  AnnotateInputDimSlots.cpp
  BufferUtils.cpp
  ComposeDimSpecs.cpp
  DimSpecsDebug.cpp
  ElideSlotPublisherAllocs.cpp
  GenerateInterface.cpp
  IdentityPropagatorRebind.cpp
  LowerAllocs.cpp
  OptimizeMemRefs.cpp
  Pipelines.cpp
  PoolAllocs.cpp
  PrintPoolStats.cpp
  PromoteStridedHipOperands.cpp
  ReservePropagatorSlots.cpp
  ResolveExternConstants.cpp
  SlotLifetimeCoalesce.cpp
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
  MLIRTensorDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  flatbuffers::flatbuffers
)
