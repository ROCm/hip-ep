##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectIR STATIC
  HipsrDialect.cpp
  HipsrShapeRegionInterface.cpp
  HipsrStartBarrierInterface.cpp
  HipsrEndBarrierInterface.cpp
  HipsrShapeYieldOp.cpp
  HipsrOps.cpp
  HipsrPoolDomainYieldOp.cpp
  HipsrPoolDomainOp.cpp
  HipsrConstantOp.cpp
  HipsrPlaceholderOp.cpp
  HipsrTraits.cpp
  HipsrCastOp.cpp
  HipsrMatMulOp.cpp
  HipsrTypes.cpp
)

add_dependencies(HipsrDialectIR
  HipsrDialectIncGen
  HipsrEnumsIncGen
  HipsrAttrsIncGen
  HipsrTypesIncGen
)

target_link_libraries(HipsrDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRShapeDialect
  # hipsr.constant owns its LLVM lowering (ConstantLowering in
  # HipsrConstantOp.cpp), so the IR lib links the LLVM conversion machinery.
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRTransforms
)
