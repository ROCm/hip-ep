##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
find_program(MLIR_PDLL_EXE mlir-pdll
    HINTS "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    NO_DEFAULT_PATH
)

if(MLIR_PDLL_EXE)
    set(PDL_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/pdl/OnnxToHipsr.pdll")
    set(PDL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/OnnxToHipsr.pdl.mlir")

    add_custom_command(
        OUTPUT "${PDL_OUTPUT}"
        COMMAND "${MLIR_PDLL_EXE}" -x mlir "${PDL_INPUT}" -o "${PDL_OUTPUT}"
        DEPENDS "${PDL_INPUT}"
        COMMENT "Compiling PDLL: OnnxToHipsr.pdll"
        VERBATIM
    )

    add_custom_target(OnnxToHipsrPDLL DEPENDS "${PDL_OUTPUT}")
    message(STATUS "OnnxToHipsr: PDLL patterns will be compiled")
endif()

# ============================================================================
# OnnxToHipsr Library
# ============================================================================

add_library(OnnxToHipsr STATIC
  ConstantConversion.cpp
  OnnxToHipsr.cpp
  CastConversion.cpp
  MatMulConversion.cpp
  ExpandConversion.cpp
  ShapeConversion.cpp
  ReshapeConversion.cpp
  UnsqueezeConversion.cpp
  EqualConversion.cpp
  TransposeConversion.cpp
  GatherConversion.cpp
  SliceConversion.cpp
  ScatterNDConversion.cpp
  ReturnConversion.cpp
)

if(MLIR_PDLL_EXE)
    add_dependencies(OnnxToHipsr OnnxToHipsrPDLL)
endif()

add_dependencies(OnnxToHipsr
  HipsrConversionPassIncGen
  HipsrDialectIncGen
  OnnxDialectIncGen
)

target_link_libraries(OnnxToHipsr PUBLIC
  HipsrDialectIR
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
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)
