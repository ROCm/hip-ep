##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
project(hip-mlir)

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

# HIP dialect static library
add_library(HipDialect STATIC HipDialect.cpp)
add_dependencies(HipDialect
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

llvm_map_components_to_libnames(llvm_libs Support Core)

target_link_libraries(HipDialect PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRDestinationStyleOpInterface
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  ${llvm_libs}
)

set_target_properties(HipDialect PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
