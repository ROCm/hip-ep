##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectPipelines STATIC
  Pipelines.cpp
)

target_link_libraries(HipsrDialectPipelines PUBLIC
  HipsrDialectTransforms
  OnnxToHipsr
  HipDialectTransforms
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRShapeOpsTransforms
  MLIRShapeToStandard
  MLIRBufferizationTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
  MLIRConvertToLLVMPass
)
