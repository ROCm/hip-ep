##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
# hip-fusion-transform matches and rewrites HIP ops through PDLL patterns
# compiled into this library as a byte array. See cmake/HipPdllPatterns.cmake
# for the toolchain discovery and the no-mlir-pdll fallback, and
# HipFusionTransform.cpp for the accessors it declares.
include(${CMAKE_SOURCE_DIR}/cmake/HipPdllPatterns.cmake)

hip_add_pdll_patterns(
  NAME HipFusionTransformPdll
  ENTRY HipFusionTransformPatterns.pdll
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pattern
  SYMBOL hip_pdl_fusion_transform
  OUT_VAR HIP_FUSION_TRANSFORM_PDL_CPP
)

add_library(HipDialectTransforms STATIC
  ${HIP_FUSION_TRANSFORM_PDL_CPP}
  AddHipContextArg.cpp
  AssignOpStateSlots.cpp
  BufferUtils.cpp
  ExternalizeConstants.cpp
  FixLoopAccumulatorOffset.cpp
  FuseROCMlir.cpp
  GenerateInterface.cpp
  GenerateOpStateInit.cpp
  HipFusionTransform.cpp
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
  ResolveMemRefDims.cpp
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

# Only defined when mlir-pdll was found; the fallback writes the byte array at
# configure time and needs no build-time step.
if(TARGET HipFusionTransformPdll)
  add_dependencies(HipDialectTransforms HipFusionTransformPdll)
endif()

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  HipConstantsIO
  HipToLLVM
  HipToTosa
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
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
  flatbuffers::flatbuffers
)
