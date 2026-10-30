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

# GPU-free numeric reference for fused QDQ LpNormalization (L2 == RMS/sqrt(N)).
add_executable(test-qlpnormalization-ref test_qlpnormalization_ref.cpp)

target_compile_features(test-qlpnormalization-ref PRIVATE cxx_std_17)

target_include_directories(test-qlpnormalization-ref PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(test-qlpnormalization-ref PRIVATE
  LLVMSupport
)

add_test(NAME QLpNormalizationRefUnit
  COMMAND test-qlpnormalization-ref
)

set_target_properties(test-qlpnormalization-ref PROPERTIES
  FOLDER "mlir/Test/Dialect/Hipsr"
)

# GPU-free check that a HIP-to-TOSA failure falls back per kernel: a static
# conv+leaky_relu stays a rocMLIR dispatch, and a fused kernel whose sub TOSA
# refuses is inlined as hip.conv, hip.sub, and hip.mul.
if(ENABLE_ROCMLIRTRITON)
  if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_rocmlir_link_closure "$<LINK_GROUP:RESCAN,${HIPDNN_ROCMLIR_LINK_LIBS}>")
  else()
    set(_rocmlir_link_closure "${HIPDNN_ROCMLIR_LINK_LIBS}")
  endif()

  foreach(_rocmlir_test IN ITEMS
      "test-rocmlir-tosa-fallback;test_rocmlir_tosa_fallback.cpp;RocMlirTosaFallbackUnit"
      "test-rocmlir-elementwise-pipeline;test_rocmlir_elementwise_pipeline.cpp;RocMlirElementwisePipelineUnit")
    list(GET _rocmlir_test 0 _target)
    list(GET _rocmlir_test 1 _source)
    list(GET _rocmlir_test 2 _ctest_name)
    add_executable(${_target} ${_source})
    target_compile_features(${_target} PRIVATE cxx_std_17)
    target_include_directories(${_target} PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_BINARY_DIR}/include
      ${MLIR_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
    )
    target_link_libraries(${_target} PRIVATE
      LibHipCompiler
      HipDialectIR
      HipDialectTransforms
      HipToTosa
      "${_rocmlir_link_closure}"
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSupport
      ${llvm_libs}
    )
    add_test(NAME ${_ctest_name} COMMAND ${_target})
    set_target_properties(${_target} PROPERTIES
      FOLDER "mlir/Test/Dialect/Hipsr"
    )
  endforeach()
endif()
