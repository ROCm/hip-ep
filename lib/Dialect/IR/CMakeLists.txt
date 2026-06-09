##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipDialect.cpp
  HipDpsOpInterface.cpp
  HipInferTypeUtils.cpp
  HipReifyResultShapesImpl.cpp
  HipShapeUtils.cpp
)

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipDpsOpInterfaceIncGen
  HipOpsIncGen
)

target_link_libraries(HipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  MLIRMemRefDialect
)

# HipDialect.cpp aggregates every TableGen-generated op class (GET_OP_CLASSES).
# As the dialect grows, this single translation unit exceeds the MSVC COFF
# section limit (fatal error C1128); /bigobj raises it.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()
