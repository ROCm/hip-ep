##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(GgmlTransforms STATIC
  ProveStateDonation.cpp
  SymbolizeShapes.cpp)
add_dependencies(GgmlTransforms GgmlTransformsPassIncGen)
target_link_libraries(GgmlTransforms PUBLIC
  GgmlDialectIR
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport)
set_target_properties(GgmlTransforms PROPERTIES
  FOLDER "mlir/Dialect"
  POSITION_INDEPENDENT_CODE ON)
