##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
project(hip-opt)

# Optional: use LLVM/MLIR from a build tree at C:/local/llvm-project
if(NOT LLVM_DIR AND EXISTS "C:/local/llvm-project/build/lib/cmake/llvm/LLVMConfig.cmake")
  set(LLVM_DIR "C:/local/llvm-project/build/lib/cmake/llvm")
endif()
if(NOT MLIR_DIR AND EXISTS "C:/local/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "C:/local/llvm-project/build/lib/cmake/mlir")
endif()

# Find LLVM using LLVM_DIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM and MLIR definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# onnx-mlir (Approach B): register the ONNX dialect for typed op matching.
# Set ONNX_MLIR_SRC and ONNX_MLIR_BUILD on the cmake command line, e.g.:
#   -DONNX_MLIR_SRC=/path/to/onnx-mlir
#   -DONNX_MLIR_BUILD=/path/to/onnx-mlir/build
# ---------------------------------------------------------------------------
if(NOT ONNX_MLIR_SRC)
  message(FATAL_ERROR
    "ONNX_MLIR_SRC is not set. Pass -DONNX_MLIR_SRC=/path/to/onnx-mlir on the cmake command line.")
endif()
if(NOT ONNX_MLIR_BUILD)
  set(ONNX_MLIR_BUILD "${ONNX_MLIR_SRC}/build")
endif()

message(STATUS "onnx-mlir source: ${ONNX_MLIR_SRC}")
message(STATUS "onnx-mlir build:  ${ONNX_MLIR_BUILD}")

include_directories(${ONNX_MLIR_SRC})
include_directories(${ONNX_MLIR_BUILD})

# Add LLVM TableGen function
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Generate dialect definition
set(LLVM_TARGET_DEFINITIONS HipDialect.td)
mlir_tablegen(HipDialect.h.inc -gen-dialect-decls -dialect=hip)
mlir_tablegen(HipDialect.cpp.inc -gen-dialect-defs -dialect=hip)
add_public_tablegen_target(HipDialectIncGen)

# Generate type definitions
set(LLVM_TARGET_DEFINITIONS HipTypes.td)
mlir_tablegen(HipTypes.h.inc -gen-typedef-decls -typedefs-dialect=hip)
mlir_tablegen(HipTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hip)
add_public_tablegen_target(HipTypesIncGen)

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS HipOps.td)
mlir_tablegen(HipOps.h.inc -gen-op-decls)
mlir_tablegen(HipOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HipOpsIncGen)

# Generate pass definitions
set(LLVM_TARGET_DEFINITIONS HipPasses.td)
mlir_tablegen(HipPasses.h.inc -gen-pass-decls -name Hip)
add_public_tablegen_target(HipPassesIncGen)

# Add the executable
add_executable(hip-opt
  hip-opt.cpp
  HipDialect.cpp
  HipToLLVM.cpp
  OnnxToHip.cpp
)

# Add dependencies on TableGen targets
add_dependencies(hip-opt
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Add hip-compiler executable
add_executable(hip-compiler
  hip-compiler.cpp
  HipDialect.cpp
  HipToLLVM.cpp
)
add_dependencies(hip-compiler
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Build static runtime library
file(GLOB RUNTIME_SRCS "ops_runtime/*.cpp")
add_library(hip_runtime_static STATIC ${RUNTIME_SRCS})
set_target_properties(hip_runtime_static PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_compile_definitions(hip_runtime_static PRIVATE __HIP_PLATFORM_AMD__ MIOPEN_BETA_API)
if(DEFINED ENV{THEROCK_DIST})
  target_include_directories(hip_runtime_static PRIVATE $ENV{THEROCK_DIST}/include)
endif()

# Link against MLIR libraries (no RegisterAll*; we register dialects explicitly)
target_link_libraries(hip-opt PRIVATE
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
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRDestinationStyleOpInterface
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
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
  ${llvm_libs}
)

target_link_libraries(hip-compiler PRIVATE
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
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransformUtils
  MLIRTargetLLVMIRExport
  MLIRExecutionEngine
  MLIRTranslateLib
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMMC
  LLVMSupport
  LLVMTarget
  LLVMIRReader
  LLVMLinker
  ${llvm_libs}
)

# ---------------------------------------------------------------------------
# onnx-mlir libraries (Approach B: typed ONNX dialect matching)
# Use linker group to handle circular deps between static archives.
# NOTE: libOMCompilerOptions.a is deliberately excluded because it registers
# an --allow-unregistered-dialect cl::opt that conflicts with MLIROptLib.
# A stub for the required symbol (disableMemRefPrefetch) is in OnnxToHip.cpp.
# ---------------------------------------------------------------------------
set(ONNX_MLIR_LIB_DIR "${ONNX_MLIR_BUILD}/Release/lib")
target_link_libraries(hip-opt PRIVATE
  -Wl,--start-group
  ${ONNX_MLIR_LIB_DIR}/libOMONNXOps.a
  ${ONNX_MLIR_LIB_DIR}/libOMMlirDialects.a
  ${ONNX_MLIR_LIB_DIR}/libOMMlirUtilities.a
  ${ONNX_MLIR_LIB_DIR}/libOMSmallFPConversion.a
  ${ONNX_MLIR_LIB_DIR}/libOMShapeHelperOpInterface.a
  ${ONNX_MLIR_LIB_DIR}/libOMONNXElementsAttr.a
  ${ONNX_MLIR_LIB_DIR}/libOMShapeInferenceOpInterface.a
  ${ONNX_MLIR_LIB_DIR}/libOMResultTypeInferenceOpInterface.a
  ${ONNX_MLIR_LIB_DIR}/libOMHasOnnxSubgraphOpInterface.a
  ${ONNX_MLIR_LIB_DIR}/libOMDiagnostic.a
  ${ONNX_MLIR_LIB_DIR}/libOMSupport.a
  ${ONNX_MLIR_LIB_DIR}/libOMAccelerator.a
  ${ONNX_MLIR_LIB_DIR}/libonnx.a
  ${ONNX_MLIR_LIB_DIR}/libonnx_proto.a
  -Wl,--end-group

  # MLIRShapeDialect must come after onnx-mlir group (onnx-mlir depends on it)
  MLIRShapeDialect
)

# protobuf is required by onnx_proto
find_package(Protobuf REQUIRED)
target_link_libraries(hip-opt PRIVATE protobuf::libprotobuf)

# Set C++ standard
set_target_properties(hip-opt hip-compiler PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
