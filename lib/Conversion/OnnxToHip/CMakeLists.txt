##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation (AMDMIGraphX approach: PDLL -> MLIR bytecode)
# ============================================================================
set(ENABLE_PDLL_FUSION ON CACHE BOOL "Enable PDLL fusion patterns (requires mlir-pdll from LLVM build)")

if(ENABLE_PDLL_FUSION)
    # Locate mlir-pdll from the LLVM build tree
    # Search order: same as AMDMIGraphX-Private
    #   1. CMAKE_BINARY_DIR/_deps/llvm-project-build/bin (our LLVM build)
    #   2. System PATH as fallback
    find_program(MLIR_PDLL_EXE mlir-pdll
        HINTS
            "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
        NO_DEFAULT_PATH
    )
    
    if(NOT MLIR_PDLL_EXE)
        message(WARNING "ENABLE_PDLL_FUSION=ON but mlir-pdll not found in LLVM build tree — "
                        "PDL fusion disabled. Build with: ninja mlir-pdll")
        set(ENABLE_PDLL_FUSION OFF CACHE BOOL "" FORCE)
    else()
        message(STATUS "OnnxToHip: mlir-pdll found: ${MLIR_PDLL_EXE}")
    endif()
endif()

if(ENABLE_PDLL_FUSION)
    set(PDL_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/pdl/QDQMatMulFusion.pdll")
    if(NOT EXISTS "${PDL_INPUT}")
        message(WARNING "OnnxToHip: QDQMatMulFusion.pdll not found at ${PDL_INPUT} — disabling PDLL")
        set(ENABLE_PDLL_FUSION OFF CACHE BOOL "" FORCE)
    else()
        message(STATUS "OnnxToHip: PDLL fusion pattern: ${PDL_INPUT}")
    endif()
endif()

if(ENABLE_PDLL_FUSION)
    set(PDL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/pdl")
    file(MAKE_DIRECTORY "${PDL_BUILD_DIR}")
    
    set(PDL_OUTPUT "${PDL_BUILD_DIR}/QDQMatMulFusion.pdl.mlir")
    
    # Compile PDLL to MLIR bytecode (not C++)
    add_custom_command(
        OUTPUT "${PDL_OUTPUT}"
        COMMAND "${MLIR_PDLL_EXE}"
                -x mlir
                "${PDL_INPUT}"
                -o "${PDL_OUTPUT}"
        DEPENDS "${PDL_INPUT}"
        COMMENT "Compiling PDLL pattern: QDQMatMulFusion.pdll -> QDQMatMulFusion.pdl.mlir"
        VERBATIM
    )
    
    add_custom_target(QDQMatMulFusionPDLL DEPENDS "${PDL_OUTPUT}")
    
    # Pass bytecode path to C++ via compile definition
    add_compile_definitions(ENABLE_PDLL_FUSION=1)
    add_compile_definitions(QDQMATMUL_FUSION_PDL_FILE="${PDL_OUTPUT}")
    
    message(STATUS "OnnxToHip: PDLL fusion enabled — bytecode: ${PDL_OUTPUT}")
else()
    message(STATUS "OnnxToHip: PDLL fusion disabled (set ENABLE_PDLL_FUSION=ON to enable)")
endif()

# ============================================================================
# OnnxToHip Library
# ============================================================================

add_library(OnnxToHip STATIC
  OnnxToHip.cpp
  SimplifyOnnx.cpp
  MatMulConversion.cpp
  TransposeConversion.cpp
  ElementwiseConversion.cpp
  PowerConversion.cpp
  ActivationConversion.cpp
  BiasGeluConversion.cpp
  FastGeluConversion.cpp
  CastConversion.cpp
  ReduceSumConversion.cpp
  ReduceMeanConversion.cpp
  ReduceL2Conversion.cpp
  MatMulNBitsConversion.cpp
  QMoEConversion.cpp
  GatherBlockQuantizedConversion.cpp
  ConvConversion.cpp
  ConvTransposeConversion.cpp
  NormConversion.cpp
  RotaryEmbeddingConversion.cpp
  OnnxRotaryEmbeddingConversion.cpp
  GqaConversion.cpp
  MultiHeadAttentionConversion.cpp
  AttentionConversion.cpp
  OnnxAttentionConversion.cpp
  HipGqaBuilder.cpp
  GatherConversion.cpp
  GatherElementsConversion.cpp
  TopKConversion.cpp
  ScatterElementsConversion.cpp
  CompressConversion.cpp
  OneHotConversion.cpp
  GatherShapeFold.cpp
  ReshapeShapeFold.cpp
  PadShapeFold.cpp
  SliceShapeFold.cpp
  ShapeConversion.cpp
  ReshapeConversion.cpp
  CausalConvWithStateConversion.cpp
  GemmConversion.cpp
  WhereConversion.cpp
  LinearAttentionConversion.cpp
  RangeConversion.cpp
  FastGeluFusion.cpp
  ErfGeluFusion.cpp
  QDQMatMulFusion.cpp
  ProjectorOpsRewrites.cpp
  LpNormalizationConversion.cpp
  EqualConversion.cpp
  DivConversion.cpp
  ReduceMaxConversion.cpp
  ReduceMinConversion.cpp
  MinConversion.cpp
  MaxConversion.cpp
  NotConversion.cpp
  CosConversion.cpp
  SinConversion.cpp
  CeilConversion.cpp
  ExpConversion.cpp
  LogConversion.cpp
  CumSumConversion.cpp
  PadConversion.cpp
  TileConversion.cpp
  ExpandConversion.cpp
  ReduceProdConversion.cpp
  LessConversion.cpp
  GreaterConversion.cpp
  GreaterOrEqualConversion.cpp
  LessOrEqualConversion.cpp
  GatherNDConversion.cpp
  SignConversion.cpp
  ModConversion.cpp
  ConstantOfShapeConversion.cpp
  SliceConversion.cpp
  ScatterNDConversion.cpp
  IdentityConversion.cpp
  OrConversion.cpp
  AndConversion.cpp
  AbsConversion.cpp
  SizeConversion.cpp
  NonZeroConversion.cpp
  ConcatConversion.cpp
  PoolConversion.cpp
  ResizeConversion.cpp
  GlobalPoolConversion.cpp
  FlattenConversion.cpp
  LoopOutline.cpp
  IfOutline.cpp
  ReluConversion.cpp
  LeakyReluConversion.cpp
  ClipConversion.cpp
)

if(ENABLE_PDLL_FUSION)
    add_dependencies(OnnxToHip QDQMatMulFusionPDLL)
endif()

add_dependencies(OnnxToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(OnnxToHip PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
