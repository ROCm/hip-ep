##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LIB_NAME mlir-imp)

add_library(${LIB_NAME}
  STATIC
  src/mlir-context-manager.hpp
  src/mlir-context-manager.cpp
  src/mlir-model.hpp
  src/mlir-model.cpp
  src/mlir-graph.hpp
  src/mlir-graph.cpp
  src/mlir-constants.hpp
  src/mlir-constants.cpp
  src/symbol-table.hpp
  src/symbol-table.cpp
  src/mlir-node-attributes.hpp
  src/mlir-node-attributes.cpp
  src/mlir-named-attribute.hpp
  src/mlir-named-attribute.cpp
  src/mlir-node-arg.hpp
  src/mlir-node-arg.cpp
  src/mlir-graph-store.hpp
  src/mlir-graph-store.cpp
  src/mlir-graph-id.hpp
  src/mlir-graph-id.cpp
  src/mlir-node-arg-index.hpp
  src/mlir-node-arg-index.cpp
  src/vaip-ort-api.cpp
  src/mlir-node.hpp
  src/mlir-node.cpp
)
target_sources(${LIB_NAME}
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/mlir.natvis>)
target_include_directories(${LIB_NAME}
  PRIVATE
  $<INSTALL_INTERFACE:include>
  src
  #${ONNX_MLIR_SOURCE_TREE}
  # ${ONNX_MLIR_BINARY_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
## enable c++17
target_compile_features(${LIB_NAME}
  PRIVATE
  cxx_std_17)
target_compile_options(${LIB_NAME}
  PRIVATE
  ${MORPHIZEN_COMPILER_OPTIONS})

target_link_libraries(${LIB_NAME}
  PRIVATE
  morphizen-utils
  vaip-ort-api-ext
  Microsoft.GSL::GSL
  # OMBuilder
  MLIRIR
  MLIRArithDialect # for mlir::arith::ConstantOp
  MLIRFuncDialect # for mlir::Func::
  MLIRParser      # for mlir::parseSourceFile
  MLIRSupport     # for MLIR support utilities
)

if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>)
endif(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
