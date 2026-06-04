##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipToLLVM STATIC
  HipToLLVM.cpp
  MemoryLowering.cpp
  ConvLowering.cpp
  ConvTransposeLowering.cpp
  MatmulLowering.cpp
  ElementwiseLowering.cpp
  PowerLowering.cpp
  ActivationLowering.cpp
  NormLowering.cpp
  GatherLowering.cpp
  RangeLowering.cpp
  CastLowering.cpp
  ReduceLowering.cpp
  TransposeLowering.cpp
  RopeLowering.cpp
  GqaLowering.cpp
  MultiHeadAttentionLowering.cpp
  MatMulNBitsLowering.cpp
  QMoELowering.cpp
  GraphLowering.cpp
  CausalConvWithStateLowering.cpp
  GemmLowering.cpp
  WhereLowering.cpp
  EqualLowering.cpp
  AndLowering.cpp
  DivLowering.cpp
  LinearAttentionLowering.cpp
  UnaryElementwiseLowering.cpp
  CumSumLowering.cpp
  PadLowering.cpp
  TileLowering.cpp
  ExpandLowering.cpp
  LessLowering.cpp
  GatherNDLowering.cpp
  ModLowering.cpp
  SliceLowering.cpp
  ScatterNDLowering.cpp
  NonZeroLowering.cpp
  SizeLowering.cpp
  LoopLowering.cpp
)

add_dependencies(HipToLLVM
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipToLLVM PUBLIC
  HipDialectIR
  MLIRPass
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRTransforms
)
