##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(TorchToHip STATIC
  TorchToHip.cpp
  MatMulConversion.cpp
  ConvConversion.cpp
  ElementwiseConversion.cpp
  ActivationConversion.cpp
  CastConversion.cpp
  GatherConversion.cpp
  ReduceConversion.cpp
  ReshapeConversion.cpp
  TransposeConversion.cpp
  NormConversion.cpp
  SliceCatConversion.cpp
)

add_dependencies(TorchToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(TorchToHip PUBLIC
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
