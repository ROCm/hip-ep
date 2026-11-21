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

# Downstream projects can use a legacy .td path as the main mlir-tblgen input.
# A forwarding include alone exits successfully but emits no C++ interface.
set(hip_build_includes "${PROJECT_BINARY_DIR}/include")
set(hip_tablegen_includes "${PROJECT_SOURCE_DIR}/include" "${hip_build_includes}"
  ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
list(REMOVE_DUPLICATES hip_tablegen_includes)
list(TRANSFORM hip_tablegen_includes PREPEND "-I")
foreach(interface IN ITEMS
    "HipDpsOpInterface;HipDpsOp"
    "HipElementMoveInterface;ElementMoveOpInterface"
    "HipOpStateInterface;OpStateOpInterface"
    "RuntimeAccessOpInterface;RuntimeAccessOpInterface")
  list(GET interface 0 td_name)
  list(GET interface 1 cpp_name)
  foreach(output IN ITEMS decls defs)
    set(test_name "HipLegacyTableGen.${td_name}.${output}")
    add_test(NAME ${test_name} COMMAND ${MLIR_TABLEGEN_EXE}
      ${hip_tablegen_includes} -gen-op-interface-${output}
      "${hip_build_includes}/hip/Dialect/IR/${td_name}.td")
    if(output STREQUAL "decls")
      set(expected "class ${cpp_name}[^a-zA-Z0-9_]")
    else()
      set(expected "mlir::hip::${cpp_name}::")
    endif()
    set_tests_properties(${test_name} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
  endforeach()
endforeach()

# Exercise the libraries a client of one HIP pass actually builds and links.
add_executable(test-hip-transforms test_hip_transforms.cpp)
target_link_libraries(test-hip-transforms PRIVATE HipDialectTransforms MLIRParser)
add_test(NAME HipTransformsStandalone COMMAND test-hip-transforms)

# A successful link alone cannot catch unused frontend archives on the link line.
# CMake 3.27+ can request the resolved target graph during configuration.
if(COMMAND cmake_file_api)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  cmake_file_api(QUERY API_VERSION 1 CODEMODEL 2)
  add_test(NAME HipTransformsDependencies
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/check_target_dependencies.py"
      "${CMAKE_BINARY_DIR}/.cmake/api/v1/reply" "$<CONFIG>"
      test-hip-transforms LibHipCompiler OnnxDialectIR OnnxToHip OnnxToHipDNN OnnxToHipsr
  )
endif()

add_executable(test-runtime-lowering-contracts
  test_runtime_lowering_contracts.cpp
)
target_include_directories(test-runtime-lowering-contracts PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Conversion/HipToLLVM
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(test-runtime-lowering-contracts PRIVATE
  HipToLLVM
  MLIRParser
)
add_test(NAME RuntimeLoweringContractsUnit
  COMMAND test-runtime-lowering-contracts
)

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

# GPU-free tests for the rocMLIR autotune candidate selection. The selection
# file has no HIP or MLIR dependency, so it compiles here directly.
add_executable(test-adaptive-benchmark
  test_adaptive_benchmark.cpp
  ${CMAKE_SOURCE_DIR}/lib/Compiler/AdaptiveBenchmark.cpp
)

target_compile_features(test-adaptive-benchmark PRIVATE cxx_std_17)

target_include_directories(test-adaptive-benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME AdaptiveBenchmarkUnit
  COMMAND test-adaptive-benchmark
)

set_target_properties(test-adaptive-benchmark PROPERTIES
  FOLDER "mlir/Test/Compiler"
)

# GPU-free check that rock.elementwise selects the elementwise pipeline.
if(ENABLE_ROCMLIRTRITON)
  if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_rocmlir_link_closure "$<LINK_GROUP:RESCAN,${HIPDNN_ROCMLIR_LINK_LIBS}>")
  else()
    set(_rocmlir_link_closure "${HIPDNN_ROCMLIR_LINK_LIBS}")
  endif()

  foreach(_rocmlir_test IN ITEMS
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
    # TritonToHsaco dlopens triton_amd_codegen by bare name (see deps.cmake).
    if(NOT WIN32)
      set_tests_properties(${_ctest_name} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib")
    endif()
    set_target_properties(${_target} PROPERTIES
      FOLDER "mlir/Test/Dialect/Hipsr"
    )
  endforeach()
endif()

# Problem-cache lookup. GPU-free: the test builds an RCAL buffer in memory
# and checks exact string hits, nearest tiles, and group mismatches.
add_executable(test-nearest-conv-tune
  test_nearest_conv_tune.cpp
  ${CMAKE_SOURCE_DIR}/tools/problem-cache/nearest_conv_tune.cpp
)

target_compile_features(test-nearest-conv-tune PRIVATE cxx_std_17)

target_include_directories(test-nearest-conv-tune PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/problem-cache
)

target_link_libraries(test-nearest-conv-tune PRIVATE HipSchemasLib)

add_test(NAME NearestConvTuneUnit COMMAND test-nearest-conv-tune)

set_target_properties(test-nearest-conv-tune PROPERTIES
  FOLDER "mlir/Test/Dialect/Hipsr"
)
