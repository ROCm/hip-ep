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
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRShapeOpsTransforms
  MLIRShapeToStandard
)
