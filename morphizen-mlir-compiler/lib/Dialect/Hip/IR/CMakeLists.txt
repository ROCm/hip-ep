##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP Dialect IR Library")

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

# Create output directory structure to match include paths
# This allows headers to use #include "morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h.inc"
set(TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR)
file(MAKE_DIRECTORY ${TABLEGEN_OUTPUT_DIR})

# Generate dialect definition
set(LLVM_TARGET_DEFINITIONS HipDialect.td)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h.inc -gen-dialect-decls -dialect=hip)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.cpp.inc -gen-dialect-defs -dialect=hip)
add_public_tablegen_target(HipDialectIncGen)

# Generate type definitions
set(LLVM_TARGET_DEFINITIONS HipTypes.td)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipTypes.h.inc -gen-typedef-decls -typedefs-dialect=hip)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hip)
add_public_tablegen_target(HipTypesIncGen)

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS HipOps.td)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipOps.h.inc -gen-op-decls)
mlir_tablegen(../../../../include/morphizen-mlir-compiler/Dialect/Hip/IR/HipOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HipOpsIncGen)

# Create HIP dialect IR library
add_library(MorphizenHipDialectIR STATIC
  HipDialect.cpp
)

# Add dependencies on TableGen targets
add_dependencies(MorphizenHipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against MLIR libraries
target_link_libraries(MorphizenHipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  ${llvm_libs}
)

# Add include directories
target_include_directories(MorphizenHipDialectIR PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Set folder for IDE organization
set_target_properties(MorphizenHipDialectIR PROPERTIES FOLDER "mlir/Dialect")
