##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation
# ============================================================================
# Locate the MLIR tools. When LLVM/MLIR is built from source as a subproject
# the tool targets exist but their .exe files are not present at configure
# time, so find_program cannot see them — use the target directly (the custom
# commands below then gain a build-time dependency on it). Fall back to
# find_program for the prebuilt / installed-LLVM case: LLVM_TOOLS_BINARY_DIR is
# exported by find_package(MLIR) and covers both the prebuilt SDK and an
# installed LLVM; the _deps path covers a FetchContent tree that has already
# been built. Default search paths stay enabled so a tool on PATH is usable.
set(PDL_TOOL_HINTS
    "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    "${LLVM_TOOLS_BINARY_DIR}"
    "${LLVM_BINARY_DIR}/bin"
    "${MLIR_BINARY_DIR}/bin"
)

if(TARGET mlir-pdll)
    set(MLIR_PDLL_EXE mlir-pdll)
    message(STATUS "OnnxToHip: using in-tree mlir-pdll target")
else()
    find_program(MLIR_PDLL_EXE NAMES mlir-pdll HINTS ${PDL_TOOL_HINTS})
endif()

# mlir-opt condenses the patterns before they are embedded. It is optional:
# without it the textual module is embedded as-is, which the parser accepts
# because it dispatches on the bytecode magic bytes.
if(TARGET mlir-opt)
    set(MLIR_OPT_EXE mlir-opt)
    message(STATUS "OnnxToHip: using in-tree mlir-opt target")
else()
    find_program(MLIR_OPT_EXE NAMES mlir-opt HINTS ${PDL_TOOL_HINTS})
endif()

# The patterns are compiled into the EP library as a byte array, so nothing
# has to be shipped beside it. ConvertOnnxToHipPass reads this translation
# unit's accessors; see its extern declarations.
set(PDL_EMBED_CPP "${CMAKE_CURRENT_BINARY_DIR}/pdl_fused_pattern_data.cpp")

if(MLIR_PDLL_EXE)
    if(NOT TARGET mlir-pdll)
        message(STATUS "OnnxToHip: mlir-pdll found: ${MLIR_PDLL_EXE}")
    endif()

    # Embedding goes through cmake/xxd.py, which needs an interpreter. Sibling
    # directories run find_package(Python3) separately because it only
    # populates the directory scope it is called from.
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
        find_program(Python3_EXECUTABLE python)
    endif()
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR
            "OnnxToHip: Python not found, so the PDL fusion patterns cannot be "
            "embedded into the EP library. The EP would build but fail to "
            "compile any model.")
    endif()

    # HipFusionPatterns.pdll #includes every pattern file, so mlir-pdll runs
    # once and emits one PDL module. Every .pdll is still globbed as a
    # dependency, otherwise editing an included pattern would not rebuild it.
    set(PDL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pdl")
    set(PDL_ENTRY "${PDL_SOURCE_DIR}/HipFusionPatterns.pdll")
    file(GLOB PDL_SOURCES CONFIGURE_DEPENDS "${PDL_SOURCE_DIR}/*.pdll")
    set(PDL_TEXT "${CMAKE_CURRENT_BINARY_DIR}/HipFusionPatterns.pdl.mlir")
    if(TARGET mlir-pdll)
        set(MLIR_PDLL_COMMAND "$<TARGET_FILE:mlir-pdll>")
        set(MLIR_PDLL_DEPENDS mlir-pdll)
    else()
        set(MLIR_PDLL_COMMAND "${MLIR_PDLL_EXE}")
        set(MLIR_PDLL_DEPENDS "${MLIR_PDLL_EXE}")
    endif()
    add_custom_command(OUTPUT "${PDL_TEXT}"
        COMMAND "${MLIR_PDLL_COMMAND}" -x mlir "-I${PDL_SOURCE_DIR}"
                "${PDL_ENTRY}" -o "${PDL_TEXT}"
        DEPENDS ${PDL_SOURCES} ${MLIR_PDLL_DEPENDS}
        COMMENT "Compiling PDLL patterns -> HipFusionPatterns.pdl.mlir"
        VERBATIM)

    if(MLIR_OPT_EXE)
        if(TARGET mlir-opt)
            set(MLIR_OPT_COMMAND "$<TARGET_FILE:mlir-opt>")
            set(MLIR_OPT_DEPENDS mlir-opt)
        else()
            set(MLIR_OPT_COMMAND "${MLIR_OPT_EXE}")
            set(MLIR_OPT_DEPENDS "${MLIR_OPT_EXE}")
            message(STATUS "OnnxToHip: mlir-opt found: ${MLIR_OPT_EXE}")
        endif()
        # mlir-pdll hardcodes enableDebugInfo(), so two thirds of its output is
        # location aliases repeating an absolute source path. Stripping them
        # also keeps the build directory out of the shipped library. Bytecode
        # then folds the remaining names into a string table.
        set(PDL_EMBED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/HipFusionPatterns.pdl.mlirbc")
        add_custom_command(OUTPUT "${PDL_EMBED_SOURCE}"
            COMMAND "${MLIR_OPT_COMMAND}" --strip-debuginfo --emit-bytecode
                    "${PDL_TEXT}" -o "${PDL_EMBED_SOURCE}"
            DEPENDS "${PDL_TEXT}" ${MLIR_OPT_DEPENDS}
            COMMENT "Condensing PDL patterns -> HipFusionPatterns.pdl.mlirbc"
            VERBATIM)
    else()
        message(STATUS
            "OnnxToHip: mlir-opt not found; embedding the textual PDL module.")
        set(PDL_EMBED_SOURCE "${PDL_TEXT}")
    endif()

    # xxd.py terminates the array with an extra 0x00, so the accessor reports
    # the real file size instead of sizeof().
    add_custom_command(OUTPUT "${PDL_EMBED_CPP}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/xxd.py"
                --var pdl_fused_pattern
                --output "${PDL_EMBED_CPP}.tmp"
                "${PDL_EMBED_SOURCE}"
        COMMAND "${Python3_EXECUTABLE}" -c
                "import os,sys; tmp,out,blob = sys.argv[1:4]; n = os.path.getsize(blob); open(out,'w').write('#include <cstddef>\\n' + open(tmp).read() + f'\\nextern \"C\" const unsigned char *hip_pdl_fused_pattern_data(void) {{ return pdl_fused_pattern; }}\\nextern \"C\" size_t hip_pdl_fused_pattern_size(void) {{ return {n}; }}\\n')"
                "${PDL_EMBED_CPP}.tmp" "${PDL_EMBED_CPP}" "${PDL_EMBED_SOURCE}"
        COMMAND "${CMAKE_COMMAND}" -E remove "${PDL_EMBED_CPP}.tmp"
        DEPENDS "${PDL_EMBED_SOURCE}"
        COMMENT "Embedding PDL patterns via pdl_fused_pattern_data.cpp"
        VERBATIM)
    add_custom_target(HipPdllPatterns DEPENDS "${PDL_EMBED_CPP}")
    message(STATUS "OnnxToHip: PDLL fusion enabled — embedding ${PDL_EMBED_SOURCE}")
else()
    # No mlir-pdll -> empty pattern blob. ConvertOnnxToHipPass treats size 0 as
    # fatal, but unrelated targets still build.
    file(WRITE "${PDL_EMBED_CPP}"
        "#include <cstddef>\n"
        "static const unsigned char pdl_fused_pattern[] = {0};\n"
        "extern \"C\" const unsigned char *hip_pdl_fused_pattern_data(void) {\n"
        "  return pdl_fused_pattern;\n"
        "}\n"
        "extern \"C\" size_t hip_pdl_fused_pattern_size(void) { return 0; }\n"
    )
    message(WARNING
        "OnnxToHip: mlir-pdll not found. ConvertOnnxToHipPass requires the "
        "compiled PDL fusion patterns embedded in the EP library.\n"
        "\n"
        "To fix:\n"
        "  - From-source LLVM: ensure mlir is in LLVM_ENABLE_PROJECTS "
        "(cmake/deps.cmake already does this) and reconfigure after LLVM is "
        "in the build tree.\n"
        "  - Prebuilt LLVM: install the MLIR tools (mlir-pdll) into the same "
        "prefix as find_package(MLIR), or put mlir-pdll on PATH.\n"
        "\n"
        "Searched in:\n"
        "  - ${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin\n"
        "  - ${LLVM_TOOLS_BINARY_DIR}\n"
        "  - ${LLVM_BINARY_DIR}/bin\n"
        "  - ${MLIR_BINARY_DIR}/bin\n"
        "  - System PATH")
endif()

# ============================================================================
# OnnxToHip Library
# ============================================================================

add_library(OnnxToHip STATIC
  "${PDL_EMBED_CPP}"
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
