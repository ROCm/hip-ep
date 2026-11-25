##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(GgmlDialectIR STATIC
  GgmlDialect.cpp
  GgmlMetadata.cpp
  GgmlOps.cpp
  ops/Elementwise.cpp
  ops/FlashAttentionExt.cpp
  ops/GetRows.cpp
  ops/Glu.cpp
  ops/MulMat.cpp
  ops/Permute.cpp
  ops/Reshape.cpp
  ops/Rope.cpp
  ops/SetRows.cpp
  ops/View.cpp)
add_dependencies(GgmlDialectIR
  GgmlAttrsIncGen
  GgmlDialectIncGen
  GgmlEnumsIncGen
  GgmlOpsIncGen
  GgmlShapeExprIncGen
  GgmlShapeRuleIncGen)
target_link_libraries(GgmlDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface)
set_target_properties(GgmlDialectIR PROPERTIES
  FOLDER "mlir/Dialect"
  POSITION_INDEPENDENT_CODE ON)
