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

# Production HIP carrier/externalizer coverage with an injected FileSystem.
# Kept as a separate target so the existing Hipsr unit remains unchanged.
add_executable(test-hip-externalize-constants
  test_hip_externalize_constants.cpp
)

target_compile_features(test-hip-externalize-constants PRIVATE cxx_std_17)

target_include_directories(test-hip-externalize-constants PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include
)

target_link_libraries(test-hip-externalize-constants PRIVATE
  HipDialectTransforms
  HipDialectIR
  HipConstantsIO
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_test(NAME HipExternalizeConstantsUnit
  COMMAND test-hip-externalize-constants
)

set_target_properties(test-hip-externalize-constants PROPERTIES
  FOLDER "mlir/Test/Dialect/Hip"
)
