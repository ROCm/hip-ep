##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# The split frontend (HIPDNN_EP_BUILD_GGML): ggml tensors captured, emitted
# as the ggml dialect. Links ggml (cmake/Ggml.cmake).

set(_ggml_incgen_targets
  GgmlSourceIncGen
  GgmlAttrsIncGen
  GgmlDialectIncGen
  GgmlEnumsIncGen
  GgmlOpsIncGen)

# The capture reads ggml's tensors and links no MLIR: `CapturedGraph.h`
# includes the generated parameter structs alone.
add_library(GgmlCapture STATIC CaptureGraph.cpp SplitGraph.cpp)
add_dependencies(GgmlCapture ${_ggml_incgen_targets})
target_include_directories(GgmlCapture PRIVATE "${HIPDNN_EP_GGML_SOURCE_DIR}/src")
target_link_libraries(GgmlCapture PUBLIC ggml-base NlohmannJson)
set_target_properties(GgmlCapture PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(GgmlEmitter STATIC GgmlEmitter.cpp)
add_dependencies(GgmlEmitter ${_ggml_incgen_targets})
target_link_libraries(GgmlEmitter PUBLIC
  GgmlCapture
  GgmlDialectIR
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport)
set_target_properties(GgmlEmitter PROPERTIES POSITION_INDEPENDENT_CODE ON)
