##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHipsr STATIC
  OnnxToHipsr.cpp
)

add_dependencies(OnnxToHipsr
  HipConversionPassIncGen
  HipsrDialectIncGen
)

target_link_libraries(OnnxToHipsr PUBLIC
  HipsrDialectIR
  MLIRPass
  MLIRTransforms
  MLIRTensorDialect
  MLIRShapeDialect
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)
