##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectTransforms STATIC
  AddContextArgPass.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertShapeToExtentPass.cpp
  ExternalizeConstants.cpp
  GenerateInterface.cpp
  HipsrPoolAllocPass.cpp
  HipsrInlineRegionsPass.cpp
  MaterializeInitTensorsPass.cpp
  PartitionPoolDomains.cpp
  PopulateShapeRegionPass.cpp
  HipsrUseOutputAllocator.cpp
)

add_dependencies(HipsrDialectTransforms
  HipsrDialectIncGen
  HipsrTransformsPassIncGen
  HipSchemas
)

target_link_libraries(HipsrDialectTransforms PUBLIC
  HipsrDialectIR
  HipConstantsIO  # writeConstantsBinToFileSystem / ConstantEntry
  HipSchemasLib
  MLIRBufferizationDialect  # BufferizableOpInterface models
  MLIRBufferizationTransforms  # BufferViewFlowAnalysis
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
  flatbuffers::flatbuffers
)
