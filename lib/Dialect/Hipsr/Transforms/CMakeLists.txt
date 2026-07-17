##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectTransforms STATIC
  PopulateShapeRegionPass.cpp
)

add_dependencies(HipsrDialectTransforms
  HipsrDialectIncGen
  HipsrTransformsPassIncGen
)

target_link_libraries(HipsrDialectTransforms PUBLIC
  HipsrDialectIR
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRShapeDialect
)
