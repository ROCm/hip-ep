##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipDialect.cpp
  HipDpsOpInterface.cpp
  HipOpStateInterface.cpp
  HipReifyResultShapesImpl.cpp
  HipShapeUtils.cpp
)

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipDpsOpInterfaceIncGen
  HipOpStateInterfaceIncGen
  HipOpsIncGen
)

# HipDialect.cpp pulls in the TableGen-generated op definitions for every Hip op
# (each now carrying the DPS/reify/infer interface bodies). On top of the
# shape-inference cascade the single translation unit exceeds MSVC's default
# COFF section limit (fatal error C1128). /bigobj raises it. No-op on non-MSVC.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()

target_link_libraries(HipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  MLIRMemRefDialect
  # OpStateOpInterface's emitOpStateConstruct helper emits LLVM-dialect calls
  # (the per-op state constructor calls woven into the op-state init function).
  MLIRLLVMDialect
)

# HipDialect.cpp aggregates every TableGen-generated op class (GET_OP_CLASSES).
# As the dialect grows, this single translation unit exceeds the MSVC COFF
# section limit (fatal error C1128); /bigobj raises it.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()
