##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# PDLL Pattern Compilation (RocMlir fusion)
# ============================================================================
# `mlir-pdll -x cpp` compiles RocMlirFusion.pdll directly into a compilable
# translation unit (RocMlirFusion.pdll.h.inc) that RocMlirFusion.cpp #includes.
# The generated code embeds the small PDL module as text and registers each
# native trampoline (declared in RocMlirFusion.h) by name — no bytecode blob,
# no manual registration.
set(ROCMLIR_PDL_TOOL_HINTS
    "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    "${LLVM_TOOLS_BINARY_DIR}"
    "${LLVM_BINARY_DIR}/bin"
    "${MLIR_BINARY_DIR}/bin")

if(TARGET mlir-pdll)
    set(ROCMLIR_PDLL_COMMAND "$<TARGET_FILE:mlir-pdll>")
    set(ROCMLIR_PDLL_DEPENDS mlir-pdll)
else()
    find_program(ROCMLIR_PDLL_EXE NAMES mlir-pdll HINTS ${ROCMLIR_PDL_TOOL_HINTS})
    if(NOT ROCMLIR_PDLL_EXE)
        message(FATAL_ERROR
            "HipDialectTransforms: mlir-pdll not found; cannot compile the "
            "RocMlir PDLL fusion patterns.")
    endif()
    set(ROCMLIR_PDLL_COMMAND "${ROCMLIR_PDLL_EXE}")
    set(ROCMLIR_PDLL_DEPENDS "${ROCMLIR_PDLL_EXE}")
endif()

set(ROCMLIR_PDL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/RocMlir")
set(ROCMLIR_PDL_ENTRY "${ROCMLIR_PDL_SOURCE_DIR}/RocMlirFusion.pdll")
file(GLOB ROCMLIR_PDL_SOURCES CONFIGURE_DEPENDS "${ROCMLIR_PDL_SOURCE_DIR}/*.pdll")
set(ROCMLIR_PDL_INC "${CMAKE_CURRENT_BINARY_DIR}/RocMlirFusion.pdll.h.inc")
add_custom_command(OUTPUT "${ROCMLIR_PDL_INC}"
    COMMAND "${ROCMLIR_PDLL_COMMAND}" -x cpp "-I${ROCMLIR_PDL_SOURCE_DIR}"
            "${ROCMLIR_PDL_ENTRY}" -o "${ROCMLIR_PDL_INC}"
    DEPENDS ${ROCMLIR_PDL_SOURCES} ${ROCMLIR_PDLL_DEPENDS}
    COMMENT "Compiling RocMlir PDLL -> RocMlirFusion.pdll.h.inc"
    VERBATIM)
add_custom_target(RocMlirPdllGen DEPENDS "${ROCMLIR_PDL_INC}")

add_library(HipDialectTransforms STATIC
  AddHipContextArg.cpp
  AssignOpStateSlots.cpp
  BufferUtils.cpp
  DecomposeConvTranspose.cpp
  ExternalizeConstants.cpp
  FixLoopAccumulatorOffset.cpp
  FuseROCMlir.cpp
  RocMlir/RocMlirFusion.cpp
  GenerateInterface.cpp
  GenerateOpStateInit.cpp
  HoistAllocSizeArith.cpp
  InferLoopBodyShapesPass.cpp
  InferShapesPass.cpp
  LoopBodyToOutParams.cpp
  LowerAllocs.cpp
  MaterializeHostScalars.cpp
  OptimizeMemRefs.cpp
  Pipelines.cpp
  PoolAllocs.cpp
  PromoteStridedHipOperands.cpp
  RelaxMultiDynExpandShape.cpp
  ResolveExternConstants.cpp
  ResolveMemRefDims.cpp
  ResolveTensorDims.cpp
  SplitDuplicateDpsInits.cpp
  UseOutputAllocator.cpp
)

add_dependencies(HipDialectTransforms
  RocMlirPdllGen
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
  HipSchemas
)

# RocMlirFusion.cpp #includes the generated RocMlirFusion.pdll.h.inc, and both
# RocMlirFusion.cpp and FuseROCMlir.cpp include "RocMlir/RocMlirFusion.h".
target_include_directories(HipDialectTransforms PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  HipConstantsIO
  HipToLLVM
  HipToTosa
  OnnxToHip
  OnnxToHipDNN
  HipSchemasLib
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFToControlFlow
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
  MLIRUBDialect
  flatbuffers::flatbuffers
)
