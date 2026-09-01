##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
# Locate mlir-pdll from the LLVM build tree (self-built, not system)
find_program(MLIR_PDLL_EXE mlir-pdll
    HINTS
        "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    NO_DEFAULT_PATH
)

if(MLIR_PDLL_EXE)
    message(STATUS "OnnxToHip: mlir-pdll found: ${MLIR_PDLL_EXE}")
    
    set(PDL_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/pdl/QDQMatMulFusion.pdll")
    set(PDL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/pdl")
    file(MAKE_DIRECTORY "${PDL_BUILD_DIR}")
    set(PDL_OUTPUT "${PDL_BUILD_DIR}/QDQMatMulFusion.pdl.mlir")
    
    # Compile PDLL to MLIR bytecode
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
    # Will be set via target_compile_definitions after library is defined: QDQMATMUL_FUSION_PDL_FILE="${PDL_OUTPUT}")
    
    message(STATUS "OnnxToHip: PDLL fusion enabled — bytecode: ${PDL_OUTPUT}")
else()
    message(WARNING "OnnxToHip: mlir-pdll not found — PDLL fusion disabled")
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
  TransposeMatMulFold.cpp
  ReshapeShapeFold.cpp
  PadShapeFold.cpp
  SliceShapeFold.cpp
  PackBroadcastTo4D.cpp
  AttentionWindowFold.cpp
  ShapeConversion.cpp
  ReshapeConversion.cpp
  CausalConvWithStateConversion.cpp
  GemmConversion.cpp
  WhereConversion.cpp
  LinearAttentionConversion.cpp
  RangeConversion.cpp
  FastGeluFusion.cpp
  ErfGeluFusion.cpp
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
  ErfConversion.cpp
  SinConversion.cpp
  CeilConversion.cpp
  RoundConversion.cpp
  AtanConversion.cpp
  FloorConversion.cpp
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
  GridSampleConversion.cpp
  GlobalPoolConversion.cpp
  FlattenConversion.cpp
  LoopOutline.cpp
  IfOutline.cpp
  ReluConversion.cpp
  LeakyReluConversion.cpp
  ClipConversion.cpp
)

if(MLIR_PDLL_EXE)
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
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)

# Set compile definition with bytecode path (must be after library is defined)
if(MLIR_PDLL_EXE)
    target_compile_definitions(OnnxToHip PRIVATE
        "QDQMATMUL_FUSION_PDL_FILE=\"${PDL_OUTPUT}\""
    )
endif()
