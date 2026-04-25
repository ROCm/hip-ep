##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHip STATIC
  OnnxToHip.cpp
  MatMulConversion.cpp
  TransposeConversion.cpp
  ElementwiseConversion.cpp
  PowerConversion.cpp
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
  ReshapeConversion.cpp
  CausalConvWithStateConversion.cpp
  GemmConversion.cpp
  UnaryElementwiseConversion.cpp
  BinaryElementwiseConversion.cpp
  SliceConversion.cpp
  Tier2ShapeConversion.cpp
  Tier3CompareConversion.cpp
  Tier5SeqConversion.cpp
  Tier6Conversion.cpp
  LstmConversion.cpp
  StftConversion.cpp
  NonzeroConversion.cpp
  ScatterNdConversion.cpp
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
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
