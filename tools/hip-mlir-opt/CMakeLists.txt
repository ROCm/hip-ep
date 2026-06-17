##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-mlir-opt hip-mlir-opt.cpp)

add_dependencies(hip-mlir-opt
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

# Static MLIR component list (used in both non-plugin and Windows-fallback paths)
set(_mlir_static_libs
  MLIROptLib
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRDestinationStyleOpInterface
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRArithTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransformUtils
)

# When BUILD_HIP_PLUGINS=ON:
#   Linux: must link the single MLIR dylib (libMLIR.so) — hidden-visibility
#     LLVM 22 static builds cannot export MLIR TypeIDs even with -rdynamic.
#   Windows (MSVC): static link works because the MSVC linker exports all
#     symbols from EXEs by default. The MLIR dylib (MLIR.lib) may not exist
#     in a component-only prebuilt; fall back to the static list if absent.
if(BUILD_HIP_PLUGINS)
  find_library(_mlir_dylib NAMES MLIR
    HINTS ${LLVM_LIBRARY_DIR} ${MLIR_LIBRARY_DIR}
    NO_DEFAULT_PATH)
  if(_mlir_dylib OR NOT WIN32)
    message(STATUS "[hip-mlir-opt] Linking MLIR dylib for plugin support")
    set(_mlir_link_libs MLIR)
  else()
    message(STATUS "[hip-mlir-opt] MLIR dylib not found — static link + symbol export (Windows)")
    set(_mlir_link_libs ${_mlir_static_libs})
  endif()
  export_executable_symbols_for_plugins(hip-mlir-opt)
else()
  set(_mlir_link_libs ${_mlir_static_libs})
endif()

target_link_libraries(hip-mlir-opt PRIVATE
  HipSupport
  LibHipCompiler
  HipDialectIR
  HipDialectTransforms
  HipToLLVM
  OnnxToHip
  ${_mlir_link_libs}
  ${llvm_libs}
)

install(TARGETS hip-mlir-opt RUNTIME DESTINATION bin)
