##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
# Locate mlir-pdll. LLVM_TOOLS_BINARY_DIR is exported by find_package(MLIR) and
# covers both the prebuilt SDK and an installed LLVM; the _deps path covers an
# LLVM built inside this project's FetchContent tree. Default search paths stay
# enabled so a tool on PATH is still usable.
find_program(MLIR_PDLL_EXE mlir-pdll
    HINTS
        "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
        "${LLVM_TOOLS_BINARY_DIR}"
        "${LLVM_BINARY_DIR}/bin"
)

if(MLIR_PDLL_EXE)
    message(STATUS "OnnxToHip: mlir-pdll found: ${MLIR_PDLL_EXE}")
    # HipFusionPatterns.pdll #includes every pattern file, so mlir-pdll runs
    # once and emits one PDL module. Every .pdll is still globbed as a
    # dependency, otherwise editing an included pattern would not rebuild it.
    set(PDL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pdl")
    file(GLOB PDL_SOURCES CONFIGURE_DEPENDS "${PDL_SOURCE_DIR}/*.pdll")
    set(PDL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/HipFusionPatterns.pdl.mlir")
    # ConvertOnnxToHipPass loads the patterns from the EP library's own
    # directory, which multi-config generators put under a per-config
    # subdirectory of CMAKE_RUNTIME_OUTPUT_DIRECTORY.
    get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(PDL_STAGED
        "${CMAKE_BINARY_DIR}/bin/$<$<BOOL:${_multi_config}>:$<CONFIG>/>HipFusionPatterns.pdl.mlir")
    add_custom_command(OUTPUT "${PDL_OUTPUT}" "${PDL_STAGED}"
        COMMAND "${MLIR_PDLL_EXE}" -x mlir "-I${PDL_SOURCE_DIR}"
                "${PDL_SOURCE_DIR}/HipFusionPatterns.pdll" -o "${PDL_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${PDL_OUTPUT}" "${PDL_STAGED}"
        DEPENDS ${PDL_SOURCES}
        COMMENT "Compiling PDLL patterns -> HipFusionPatterns.pdl.mlir"
        VERBATIM)
    add_custom_target(HipPdllPatterns DEPENDS "${PDL_OUTPUT}" "${PDL_STAGED}")
    install(FILES "${PDL_OUTPUT}" DESTINATION bin)
    # Every packaging path (install tree, wheel, staged test package) has to
    # ship this file beside the EP library. Export it for those consumers.
    set(HIP_PDL_FUSION_PATTERNS "${PDL_OUTPUT}" CACHE INTERNAL
        "Compiled PDL fusion patterns that must ship beside the EP library")
    message(STATUS "OnnxToHip: PDLL fusion enabled — patterns: ${PDL_OUTPUT}")
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
  QMoEAmdConversion.cpp
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
  QdqConversion.cpp
  LoopOutline.cpp
  IfOutline.cpp
  ReluConversion.cpp
  LeakyReluConversion.cpp
  SwishConversion.cpp
  ClipConversion.cpp
)

if(TARGET HipPdllPatterns)
    add_dependencies(OnnxToHip HipPdllPatterns)
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
