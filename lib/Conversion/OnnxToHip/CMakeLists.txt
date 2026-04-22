##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHip STATIC
  OnnxToHip.cpp
  MatMulConversion.cpp
  TransposeConversion.cpp
  ElementwiseConversion.cpp
  ActivationConversion.cpp
  CastConversion.cpp
  ReduceSumConversion.cpp
  MatMulNBitsConversion.cpp
  QMoEConversion.cpp
  ConvConversion.cpp
  NormConversion.cpp
  RotaryEmbeddingConversion.cpp
  GqaConversion.cpp
  GatherConversion.cpp
  CompileTimeOpsConversion.cpp
  ReshapeConversion.cpp
  GemmConversion.cpp
)

add_dependencies(OnnxToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(OnnxToHip PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
