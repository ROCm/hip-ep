##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectIR STATIC
  HipsrDialect.cpp
  HipsrShapeYieldOp.cpp
  HipsrOps.cpp
  HipsrAllocOutputOp.cpp
  HipsrPoolDomainYieldOp.cpp
  HipsrPoolDomainOp.cpp
  HipsrComputeYieldOp.cpp
  HipsrComputeOp.cpp
  HipsrConstantOp.cpp
  HipsrPlaceholderOp.cpp
  HipsrCastOp.cpp
  HipsrMatMulOp.cpp
  HipsrExpandOp.cpp
  HipsrAddOp.cpp
  HipsrMulOp.cpp
  HipsrEqualOp.cpp
  HipsrTransposeOp.cpp
  HipsrGetPoolOp.cpp
  HipsrPreserveShapeOp.cpp
  HipsrLLVMLoweringUtils.cpp
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
  MLIRControlFlowInterfaces
  MLIRDestinationStyleOpInterface
  MLIRSideEffectInterfaces
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRShapeDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRConvertToLLVMInterface
  MLIRTransforms
  MLIRTensorDialect
)
