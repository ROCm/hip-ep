##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrToLLVM STATIC
  HipsrToLLVM.cpp
  ConstantLowering.cpp
)

add_dependencies(HipsrToLLVM
  HipsrConversionPassIncGen
  HipsrDialectIncGen
  HipsrEnumsIncGen
  HipsrAttrsIncGen
  HipsrTypesIncGen
  HipDialectIncGen
  HipTypesIncGen
)

target_link_libraries(HipsrToLLVM PUBLIC
  HipsrDialectIR
  HipDialectIR
  MLIRPass
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRConvertToLLVMInterface
  MLIRMemRefDialect
  MLIRTransforms
  MLIRArithTransforms
  MLIRIR
  MLIRSupport
)
