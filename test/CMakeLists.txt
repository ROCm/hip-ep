##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ROCm libraries configuration
set(THEROCK_DIST "" CACHE PATH "TheRock distribution path")

if(THEROCK_DIST)
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
endif()

find_package(hip QUIET)
find_package(miopen QUIET)
find_package(hipblaslt QUIET)

if(NOT hip_FOUND OR NOT miopen_FOUND OR NOT hipblaslt_FOUND)
  message(STATUS "ROCm libraries not found, skipping ROCm tests")
  return()
endif()

message(STATUS "Building ROCm tests")

# Find GTest for the tests
find_package(GTest REQUIRED)

# Conv test (MIOpen)
add_executable(rocm_conv_test test_conv.cpp)
target_link_libraries(rocm_conv_test PRIVATE MIOpen GTest::gtest GTest::gtest_main)
if(WIN32)
  target_include_directories(rocm_conv_test SYSTEM PRIVATE "${THEROCK_DIST}/include")
  target_compile_definitions(rocm_conv_test PRIVATE "__HIP_PLATFORM_AMD__=1")
  target_link_libraries(rocm_conv_test PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  target_link_libraries(rocm_conv_test PRIVATE hip::device)
endif()
set_target_properties(rocm_conv_test PROPERTIES FOLDER "rocm-tests")

# Gemm test (hipBLASLt)
add_executable(rocm_gemm_test test_gemm.cpp)
target_link_libraries(rocm_gemm_test PRIVATE roc::hipblaslt GTest::gtest GTest::gtest_main)
if(WIN32)
  target_include_directories(rocm_gemm_test SYSTEM PRIVATE "${THEROCK_DIST}/include")
  target_compile_definitions(rocm_gemm_test PRIVATE "__HIP_PLATFORM_AMD__=1")
  target_link_libraries(rocm_gemm_test PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  target_link_libraries(rocm_gemm_test PRIVATE hip::device)
endif()
set_target_properties(rocm_gemm_test PROPERTIES FOLDER "rocm-tests")

# Timeout test (GPU timeout mechanism)
add_executable(rocm_timeout_test test_timeout.cpp)
target_link_libraries(rocm_timeout_test PRIVATE MIOpen glog::glog)
if(WIN32)
  target_include_directories(rocm_timeout_test SYSTEM PRIVATE 
    "${THEROCK_DIST}/include"
    "${CMAKE_SOURCE_DIR}/../onnxruntime/include/onnxruntime/core/session"
  )
  target_compile_definitions(rocm_timeout_test PRIVATE "__HIP_PLATFORM_AMD__=1")
  target_link_libraries(rocm_timeout_test PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  target_link_libraries(rocm_timeout_test PRIVATE hip::device)
endif()
set_target_properties(rocm_timeout_test PROPERTIES FOLDER "rocm-tests")

# ORT Integration test (for VitisAI EP testing with MY_LOG)
# This test links against ONNX Runtime and can trigger the passes/custom-ops

# Check for ORT in the source tree (from parent project)
set(ORT_SOURCE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime/include/onnxruntime/core/session")
set(ORT_BUILD_LIB "${CMAKE_BINARY_DIR}/bin/onnxruntime.dll")
# ORT import library is in the ORT build directory (C:/Develop/m/build/onnxruntime/Release)
# The path from source dir to build dir goes up to m/ then to build/
set(ORT_BUILD_IMPLIB "${CMAKE_SOURCE_DIR}/../../build/onnxruntime/Release/onnxruntime.lib")

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
  set_target_properties(ort_integration_test PROPERTIES FOLDER "rocm-tests")
elseif(ORT_SOURCE_INCLUDE_DIR AND EXISTS "${ORT_BUILD_IMPLIB}")
  # Use ORT from build output (common case when building morphizen-rocm)
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
  set_target_properties(ort_integration_test PROPERTIES FOLDER "rocm-tests")
elseif(ORT_SOURCE_INCLUDE_DIR)
  # ORT headers found but no import lib - create test that will need DLL at runtime
  message(STATUS "Building ORT integration test (headers only - needs DLL at runtime)")
  message(STATUS "  ORT include: ${ORT_SOURCE_INCLUDE_DIR}")
  
  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    GTest::gtest
    GTest::gtest_main
  )
  target_include_directories(ort_integration_test PRIVATE "${ORT_SOURCE_INCLUDE_DIR}")
  target_compile_features(ort_integration_test PRIVATE cxx_std_17)
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
    # Link against the DLL in bin folder
    target_link_directories(ort_integration_test PRIVATE "${CMAKE_BINARY_DIR}/bin")
    target_link_libraries(ort_integration_test PRIVATE onnxruntime)
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "rocm-tests")
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
    set_target_properties(ort_integration_test PROPERTIES FOLDER "rocm-tests")
  else()
    message(STATUS "ONNX Runtime not found, skipping ORT integration test")
  endif()
endif()
