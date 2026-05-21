##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipConversionCommon STATIC
  DimSpecMaterializer.cpp
)

add_dependencies(HipConversionCommon
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

target_include_directories(HipConversionCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(HipConversionCommon PUBLIC
  HipDialectIR
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
