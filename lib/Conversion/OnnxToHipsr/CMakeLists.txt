##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
# Locate mlir-pdll from the LLVM build tree
find_program(MLIR_PDLL_EXE mlir-pdll
    HINTS
        "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    NO_DEFAULT_PATH
)

if(MLIR_PDLL_EXE)
    message(STATUS "OnnxToHipsr: mlir-pdll found: ${MLIR_PDLL_EXE}")

    set(PDL_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/pdl/OnnxToHipsr.pdll")
    set(PDL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/OnnxToHipsr.pdl.mlir")

    # Compile PDLL to MLIR bytecode
    add_custom_command(
        OUTPUT "${PDL_OUTPUT}"
        COMMAND "${MLIR_PDLL_EXE}"
                -x mlir
                "${PDL_INPUT}"
                -o "${PDL_OUTPUT}"
        DEPENDS "${PDL_INPUT}"
        COMMENT "Compiling PDLL pattern: OnnxToHipsr.pdll"
        VERBATIM
    )

    add_custom_target(OnnxToHipsrPDLL DEPENDS "${PDL_OUTPUT}")

    message(STATUS "OnnxToHipsr: PDLL fusion enabled — bytecode: ${PDL_OUTPUT}")
else()
    message(WARNING "OnnxToHipsr: mlir-pdll not found — PDLL conversion disabled")
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

# Pass bytecode path to C++ via compile definition
if(MLIR_PDLL_EXE)
    target_compile_definitions(OnnxToHipsr PRIVATE
        "ONNX_TO_HIPSR_PDL_FILE=\"${PDL_OUTPUT}\""
    )
endif()
