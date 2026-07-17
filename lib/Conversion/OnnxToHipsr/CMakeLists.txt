##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHipsr STATIC
  ConstantConversion.cpp
  OnnxToHipsr.cpp
  CastConversion.cpp
)

add_dependencies(OnnxToHipsr
  HipsrConversionPassIncGen
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
