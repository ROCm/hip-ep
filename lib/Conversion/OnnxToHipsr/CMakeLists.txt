##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHipsr STATIC
  ConstantConversion.cpp
  OnnxToHipsr.cpp
  CastConversion.cpp
  MatMulConversion.cpp
  ExpandConversion.cpp
  ShapeConversion.cpp
  ReshapeConversion.cpp
  UnsqueezeConversion.cpp
  EqualConversion.cpp
  TransposeConversion.cpp
  GatherConversion.cpp
  SliceConversion.cpp
  ScatterNDConversion.cpp
  NonZeroConversion.cpp
  ReturnConversion.cpp
)

add_dependencies(OnnxToHipsr
  HipsrConversionPassIncGen
  HipsrDialectIncGen
)

target_link_libraries(OnnxToHipsr PUBLIC
  HipsrDialectIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRShapeDialect
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)
