##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectTransforms STATIC
  ExternalizeConstants.cpp
  HipsrPoolAllocPass.cpp
  PartitionPoolDomains.cpp
  PopulateShapeRegionPass.cpp
)

add_dependencies(HipsrDialectTransforms
  HipsrDialectIncGen
  HipsrTransformsPassIncGen
)

target_link_libraries(HipsrDialectTransforms PUBLIC
  HipsrDialectIR
  HipConstantsIO  # writeConstantsBinToFileSystem / ConstantEntry
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)
