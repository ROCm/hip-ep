##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectIR STATIC
  HipsrDialect.cpp
  HipsrStartBarrierInterface.cpp
  HipsrEndBarrierInterface.cpp
  HipsrShapeYieldOp.cpp
  HipsrOps.cpp
  HipsrPoolDomainYieldOp.cpp
  HipsrPoolDomainOp.cpp
  HipsrComputeYieldOp.cpp
  HipsrComputeOp.cpp
  HipsrEmptyYieldOp.cpp
  HipsrEmptyOp.cpp
  HipsrConstantOp.cpp
  HipsrPlaceholderOp.cpp
  HipsrCastOp.cpp
  HipsrMatMulOp.cpp
  HipsrExpandOp.cpp
  HipsrAddOp.cpp
  HipsrGetPoolOp.cpp
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
