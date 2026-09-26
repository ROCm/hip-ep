##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHipsr STATIC
  ConstantConversion.cpp
  OnnxToHipsr.cpp
  ConversionInScheme.cpp
  CastConversion.cpp
  MatMulConversion.cpp
  ExpandConversion.cpp
  MinConversion.cpp
  ShapeConversion.cpp
  ReshapeConversion.cpp
  UnsqueezeConversion.cpp
  EqualConversion.cpp
  TransposeConversion.cpp
  GatherConversion.cpp
  SliceConversion.cpp
  ScatterNDConversion.cpp
  NonZeroConversion.cpp
  ReturnConversion.cpp
)

add_dependencies(OnnxToHipsr
  HipsrConversionPassIncGen
  HipsrDialectIncGen
  OnnxDialectIncGen
)

target_link_libraries(OnnxToHipsr PUBLIC
  HipsrDialectIR
  HipsrSchemeRuntime
  OnnxDialectIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRTensorDialect
  MLIRShapeDialect
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)

# Define Scheme source/build directories for development
# Enable runtime compilation: source .sls → binary .so in build dir
# CMAKE_CURRENT_SOURCE_DIR points to lib/Conversion/OnnxToHipsr
get_filename_component(SCHEME_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Hipsr/Scheme/libraries"
  ABSOLUTE)
get_filename_component(RIME_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/rime"
  ABSOLUTE)
set(SCHEME_BUILD_DIR "${CMAKE_BINARY_DIR}/lib/scheme/libraries")
set(RIME_BUILD_DIR "${CMAKE_BINARY_DIR}/lib/scheme/rime")

target_compile_definitions(OnnxToHipsr PRIVATE
  CHEZ_SCHEME_SOURCE_DIR="${SCHEME_SOURCE_DIR}"
  CHEZ_SCHEME_BUILD_DIR="${SCHEME_BUILD_DIR}"
  CHEZ_RIME_SOURCE_DIR="${RIME_SOURCE_DIR}"
  CHEZ_RIME_BUILD_DIR="${RIME_BUILD_DIR}"
)
