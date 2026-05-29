##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipDialect.cpp
  HipShapeUtils.cpp
)

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
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
