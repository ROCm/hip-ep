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

# Add the executable
add_executable(hip-opt 
  hip-opt.cpp
  HipDialect.cpp
  HipToLLVM.cpp
)

# Add dependencies on TableGen targets
add_dependencies(hip-opt 
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

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
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRTransformUtils
  ${llvm_libs}
)

# Set C++ standard
set_target_properties(hip-opt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
