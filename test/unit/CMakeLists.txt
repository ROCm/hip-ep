##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# MLIR dialect/pass unit tests (GPU-free).
#=============================================================================
# Cover behavior that LIT cannot: here -hipsr-externalize-constants runs with an
# injected in-memory FileSystem so the Phase 2 constants-file write (and the bytes it
# emits) can be asserted, which hip-mlir-opt (no FileSystem) cannot exercise.
# Plain main() (no GTest), matching the other MLIR-side unit tests.

message(STATUS "Configuring MLIR dialect/pass unit tests")

add_executable(test-hipsr-externalize-constants test_externalize_constants.cpp)

target_compile_features(test-hipsr-externalize-constants PRIVATE cxx_std_17)

target_include_directories(test-hipsr-externalize-constants PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include  # file_io.hpp
)

target_link_libraries(test-hipsr-externalize-constants PRIVATE
  HipsrDialectTransforms
  HipsrDialectIR
  HipConstantsIO
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRSupport
)

add_test(NAME HipsrExternalizeConstantsUnit
  COMMAND test-hipsr-externalize-constants
)

set_target_properties(test-hipsr-externalize-constants PROPERTIES
  FOLDER "mlir/Test/Dialect/Hipsr"
)

#=============================================================================
# ONNX-dtype -> MLIR-type mapping unit test (GPU-free).
#=============================================================================
# Covers onnxElementTypeToMlirDenseElementType (morphizen/mlir-imp), which LIT
# cannot reach: hip-mlir-opt has no ONNX front end and only consumes textual
# MLIR. Regression guard for the QDQ UINT16 sign-flip (inline UINT16 dense
# constants must stay ui16, not signless i16). Plain main() (no GTest).

add_executable(test-dtype-mapping test_dtype_mapping.cpp)

target_compile_features(test-dtype-mapping PRIVATE cxx_std_17)

target_include_directories(test-dtype-mapping PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/morphizen/mlir-imp/src  # mlir-constants.hpp
)

target_link_libraries(test-dtype-mapping PRIVATE
  mlir-imp
  MLIRIR
  MLIRSupport
)

add_test(NAME MlirDtypeMappingUnit
  COMMAND test-dtype-mapping
)

set_target_properties(test-dtype-mapping PROPERTIES
  FOLDER "mlir/Test/Dialect/Hipsr"
)
