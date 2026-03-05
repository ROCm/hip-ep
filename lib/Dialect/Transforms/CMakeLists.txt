##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generate Passes.h.inc from include/hip/Dialect/Transforms/Passes.td
set(TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include/hip/Dialect/Transforms)
file(RELATIVE_PATH TABLEGEN_OUTPUT_REL
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${TABLEGEN_OUTPUT_DIR}"
)
file(MAKE_DIRECTORY ${TABLEGEN_OUTPUT_DIR})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/hip/Dialect/Transforms/Passes.td)
mlir_tablegen(${TABLEGEN_OUTPUT_REL}/Passes.h.inc -gen-pass-decls -name Hip)
add_public_tablegen_target(HipPassesIncGen)

add_library(HipDialectTransforms STATIC
  OptimizeMemRefs.cpp
  PoolAllocs.cpp
)

add_dependencies(HipDialectTransforms
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
)

target_include_directories(HipDialectTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)
