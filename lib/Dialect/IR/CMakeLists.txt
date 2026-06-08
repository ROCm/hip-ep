##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipDialect.cpp
  HipDpsOpInterface.cpp
  HipReifyResultShapesImpl.cpp
  HipShapeUtils.cpp
)

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipDpsOpInterfaceIncGen
  HipOpsIncGen
)

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
)
