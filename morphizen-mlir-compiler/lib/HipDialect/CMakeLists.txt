##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP MLIR Dialect Library")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Set TableGen executables with absolute paths
# Visual Studio MSBuild doesn't inherit PATH, so we need absolute paths to mlir-tblgen/llvm-tblgen.
# Try LLVM_TOOLS_BINARY_DIR first (FetchContent), then CMAKE_INSTALL_PREFIX/bin (pre-installed).
if(EXISTS "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  set(MLIR_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(MLIR_TABLEGEN_EXE "${CMAKE_INSTALL_PREFIX}/bin/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
endif()

if(EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  set(LLVM_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(LLVM_TABLEGEN_EXE "${CMAKE_INSTALL_PREFIX}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
endif()

message(STATUS "Found mlir-tblgen: ${MLIR_TABLEGEN_EXE}")
message(STATUS "Found llvm-tblgen: ${LLVM_TABLEGEN_EXE}")

include(TableGen)

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

# Create HIP dialect library (renamed with Morphizen prefix)
add_library(MorphizenHipDialect STATIC
  HipDialect.cpp
  HipToLLVM.cpp
  OnnxToHip.cpp
  GenerateInterfacePass.cpp
  MemoryPoolingPass.cpp
  HipBufferDeallocationPass.cpp
)

# Add dependencies on TableGen targets
add_dependencies(MorphizenHipDialect
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against MLIR libraries and onnx-mlir
target_link_libraries(MorphizenHipDialect PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  # MLIR conversion libraries (needed for HipToLLVM lowering)
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  OMONNXOps  # ONNX dialect from our 3rd-party/onnx-mlir
  ${llvm_libs}
)

# Add onnx-mlir include directories (local to morphizen-mlir-compiler)
# The code uses: #include "src/Dialect/ONNX/ONNXOps.hpp"
# So we need to add the directory ABOVE 'src' to the include path
target_include_directories(MorphizenHipDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR source headers
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir   # ONNX-MLIR generated headers (.inc files)
)

message(STATUS "ONNX-MLIR source: ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir")
message(STATUS "ONNX-MLIR build: ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir")

message(STATUS "MorphizenHipDialect: Linked with onnx-mlir ONNX dialect")

# Set folder for IDE organization
set_target_properties(MorphizenHipDialect PROPERTIES FOLDER "mlir")
