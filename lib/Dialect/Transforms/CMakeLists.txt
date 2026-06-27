##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectTransforms STATIC
  AddHipContextArg.cpp
  AssignOpStateSlots.cpp
  BufferUtils.cpp
  FixLoopAccumulatorOffset.cpp
  GenerateInterface.cpp
  GenerateOpStateInit.cpp
  HoistAllocSizeArith.cpp
  InferLoopBodyShapesPass.cpp
  InferShapesPass.cpp
  LoopBodyToOutParams.cpp
  LowerAllocs.cpp
  MaterializeHostScalars.cpp
  OptimizeMemRefs.cpp
  Pipelines.cpp
  PoolAllocs.cpp
  PromoteStridedHipOperands.cpp
  RelaxMultiDynExpandShape.cpp
  ResolveExternConstants.cpp
  ResolveTensorDims.cpp
  SplitDuplicateDpsInits.cpp
  UseOutputAllocator.cpp
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
  MLIRAffineDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFToControlFlow
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  flatbuffers::flatbuffers
)
