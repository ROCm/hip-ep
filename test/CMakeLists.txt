##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building MLIR tests")

# Find GTest for the tests
find_package(GTest REQUIRED)

# ORT Integration test (for VitisAI EP testing with MLIR pass)
# This test links against ONNX Runtime and can trigger the MLIR pass

# Check for ORT in the source tree (from parent project)
set(ORT_SOURCE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime/include/onnxruntime/core/session")
set(ORT_BUILD_LIB "${CMAKE_BINARY_DIR}/bin/onnxruntime.dll")
# ORT import library is in the installation directory (CMAKE_INSTALL_PREFIX/lib)
set(ORT_BUILD_IMPLIB "${CMAKE_INSTALL_PREFIX}/lib/onnxruntime.lib")

# Also check for pre-built ORT from install
if(NOT EXISTS "${ORT_SOURCE_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  set(ORT_SOURCE_INCLUDE_DIR "")
endif()

if(TARGET onnxruntime)
  message(STATUS "Building ORT integration test (using onnxruntime target)")
  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    onnxruntime
    GTest::gtest
    GTest::gtest_main
  )
  target_include_directories(ort_integration_test PRIVATE
    ${ONNX_RUNTIME_SOURCE_TREE_DIR}/include/onnxruntime/core/session
  )
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
elseif(ORT_SOURCE_INCLUDE_DIR AND EXISTS "${ORT_BUILD_IMPLIB}")
  # Use ORT from build output (common case when building morphizen-mlir)
  message(STATUS "Building ORT integration test (using build output)")
  message(STATUS "  ORT include: ${ORT_SOURCE_INCLUDE_DIR}")
  message(STATUS "  ORT lib: ${ORT_BUILD_IMPLIB}")
  
  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    "${ORT_BUILD_IMPLIB}"
    GTest::gtest
    GTest::gtest_main
  )
  target_include_directories(ort_integration_test PRIVATE "${ORT_SOURCE_INCLUDE_DIR}")
  target_compile_features(ort_integration_test PRIVATE cxx_std_17)
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
elseif(ORT_SOURCE_INCLUDE_DIR)
  # ORT headers found - use installation lib path
  message(STATUS "Building ORT integration test (using installation lib)")
  message(STATUS "  ORT include: ${ORT_SOURCE_INCLUDE_DIR}")
  message(STATUS "  ORT lib: ${ORT_BUILD_IMPLIB}")
  
  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    "${ORT_BUILD_IMPLIB}"
    GTest::gtest
    GTest::gtest_main
  )
  target_include_directories(ort_integration_test PRIVATE "${ORT_SOURCE_INCLUDE_DIR}")
  target_compile_features(ort_integration_test PRIVATE cxx_std_17)
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
else()
  # Try to find pre-installed ORT
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      ${CMAKE_PREFIX_PATH}/include/onnxruntime
      ${CMAKE_PREFIX_PATH}/include
      $ENV{ONNXRUNTIME_ROOT}/include
  )
  find_library(ORT_LIB onnxruntime
    HINTS
      ${CMAKE_PREFIX_PATH}/lib
      $ENV{ONNXRUNTIME_ROOT}/lib
  )
  
  if(ORT_INCLUDE_DIR AND ORT_LIB)
    message(STATUS "Building standalone ORT integration test")
    message(STATUS "  ORT include: ${ORT_INCLUDE_DIR}")
    message(STATUS "  ORT lib: ${ORT_LIB}")
    
    add_executable(ort_integration_test test_ort_integration.cpp)
    target_link_libraries(ort_integration_test PRIVATE
      ${ORT_LIB}
      GTest::gtest
      GTest::gtest_main
    )
    target_include_directories(ort_integration_test PRIVATE ${ORT_INCLUDE_DIR})
    if(WIN32)
      target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
    endif()
    set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
  else()
    message(STATUS "ONNX Runtime not found, skipping ORT integration test")
  endif()
endif()
