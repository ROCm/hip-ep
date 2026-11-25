##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(GgmlToHip STATIC
  AddRmsNormQ8Fusion.cpp
  BasicConversion.cpp
  ComputeConversion.cpp
  FlashAttentionFusion.cpp
  GgmlToHip.cpp
  KQuantSwiGluFusion.cpp
  LayoutConversion.cpp
  MatmulConversion.cpp
  RmsNormScaleFusion.cpp)
add_dependencies(GgmlToHip
  GgmlToHipPassIncGen
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen)
target_link_libraries(GgmlToHip PUBLIC
  GgmlDialectIR
  GgmlTransforms
  HipDialectIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms)
set_target_properties(GgmlToHip PROPERTIES
  FOLDER "mlir/Conversion"
  POSITION_INDEPENDENT_CODE ON)
