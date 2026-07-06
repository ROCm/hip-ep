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
  GatherBlockQuantizedLowering.cpp
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
  ReadbackDimLowering.cpp
  ReadbackScalarLowering.cpp
  SizeLowering.cpp
  LoopLowering.cpp
  PoolLowering.cpp
  ResizeLowering.cpp
  GlobalPoolLowering.cpp
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
  MLIRArithTransforms
  # ConvertToLLVMPatternInterface: convert-hip-to-llvm walks the module and
  # collects HIP->LLVM lowerings from out-of-tree (plugin) dialects that
  # implement this interface. See the guarded walk in HipToLLVM.cpp.
  MLIRConvertToLLVMInterface
)
