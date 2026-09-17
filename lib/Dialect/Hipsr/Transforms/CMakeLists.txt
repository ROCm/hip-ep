##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectTransforms STATIC
  AddContextArgPass.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertShapeToExtentPass.cpp
  ExternalizeConstants.cpp
  HipsrPoolAllocPass.cpp
  HipsrInlineRegionsPass.cpp
  MainGraphAbiPass.cpp
  MaterializeInitTensorsPass.cpp
  PartitionPoolDomains.cpp
  PopulateShapeRegionPass.cpp
  RecordGraphIOMetadataPass.cpp
  HipsrUseOutputAllocator.cpp
)

add_dependencies(HipsrDialectTransforms
  HipsrDialectIncGen
  HipsrTransformsPassIncGen
)

target_link_libraries(HipsrDialectTransforms PUBLIC
  HipsrDialectIR
  HipConstantsIO  # writeConstantsBinToFileSystem / ConstantEntry
  MLIRBufferizationDialect  # BufferizableOpInterface models
  MLIRBufferizationTransforms  # BufferViewFlowAnalysis
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)
