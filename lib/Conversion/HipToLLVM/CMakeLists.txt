##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipToLLVM STATIC
  HipToLLVM.cpp
  MemoryLowering.cpp
  ConvLowering.cpp
  MatmulLowering.cpp
  ElementwiseLowering.cpp
  ActivationLowering.cpp
  NormLowering.cpp
  GatherLowering.cpp
  CastLowering.cpp
  ReduceSumLowering.cpp
  TransposeLowering.cpp
  RopeLowering.cpp
  GqaLowering.cpp
  MatMulNBitsLowering.cpp
  QMoELowering.cpp
  GraphLowering.cpp
  CausalConvWithStateLowering.cpp
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
