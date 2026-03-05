##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# LIT tests (from UDNA-compiler)
#=============================================================================
if(ONNX_HIP_INCLUDE_LIT_TESTS)
    add_subdirectory(lit)
endif()

#=============================================================================
# E2E tests (includes ORT integration test from UDNA-compiler)
#=============================================================================
add_subdirectory(e2e)

#=============================================================================
# Standalone mode: Build only test_gqa without onnx-hipdnn-ep dependencies
# Usage: cmake -B build -S . -DCMAKE_PREFIX_PATH=".../local" -DSTANDALONE_BUILD=ON
#=============================================================================
option(STANDALONE_BUILD "Build test_gqa standalone (no onnx-hipdnn-ep)" OFF)

if(STANDALONE_BUILD)
  cmake_minimum_required(VERSION 3.20)
  project(test_gqa_standalone LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  # Find ONNX Runtime
  find_package(onnxruntime CONFIG REQUIRED)

  add_executable(test_gqa test_gqa.cpp)
  target_link_libraries(test_gqa PRIVATE onnxruntime::onnxruntime)

  if(WIN32)
    target_compile_definitions(test_gqa PRIVATE "_CRT_SECURE_NO_WARNINGS")

    # Copy ORT DLLs to output directory
    add_custom_command(
      TARGET test_gqa POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:onnxruntime::onnxruntime>
        $<TARGET_FILE_DIR:test_gqa>
    )
  endif()

  install(TARGETS test_gqa RUNTIME DESTINATION bin)
  message(STATUS "Standalone test_gqa configured successfully")
  return()
endif()

#=============================================================================
# test_gqa executable for GQA layer testing (standalone, no ROCm dependency)
#=============================================================================
option(BUILD_TEST_GQA "Build test_gqa executable" ON)

if(BUILD_TEST_GQA)
  message(STATUS "Building test_gqa: ENABLED")

  # Find ONNX Runtime
  find_package(onnxruntime CONFIG QUIET)

  # Also check for ORT in the source tree
  set(ORT_SOURCE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime/include/onnxruntime/core/session")
  set(ORT_BUILD_IMPLIB "${CMAKE_INSTALL_PREFIX}/lib/onnxruntime.lib")

  if(NOT EXISTS "${ORT_SOURCE_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    set(ORT_SOURCE_INCLUDE_DIR "")
  endif()

  # Try to find pre-installed ORT if not found via config
  if(NOT TARGET onnxruntime::onnxruntime)
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
  endif()

  if(TARGET onnxruntime::onnxruntime OR ORT_LIB)
    add_executable(test_gqa test_gqa.cpp)

    if(TARGET onnxruntime::onnxruntime)
      target_link_libraries(test_gqa PRIVATE onnxruntime::onnxruntime)
    else()
      target_link_libraries(test_gqa PRIVATE ${ORT_LIB})
      target_include_directories(test_gqa PRIVATE ${ORT_INCLUDE_DIR})
    endif()

    # Also link with ORT from source tree if available
    if(ORT_SOURCE_INCLUDE_DIR)
      target_include_directories(test_gqa PRIVATE "${ORT_SOURCE_INCLUDE_DIR}")
    endif()

    target_compile_features(test_gqa PRIVATE cxx_std_17)

    if(WIN32)
      target_compile_definitions(test_gqa PRIVATE "_CRT_SECURE_NO_WARNINGS")

      # Copy ORT DLLs to output directory
      if(TARGET onnxruntime::onnxruntime)
        add_custom_command(
          TARGET test_gqa POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:onnxruntime::onnxruntime>
            $<TARGET_FILE_DIR:test_gqa>
        )
      endif()
    endif()

    set_target_properties(test_gqa PROPERTIES FOLDER "gqa-tests")

    install(TARGETS test_gqa RUNTIME DESTINATION bin)

    message(STATUS "test_gqa configured successfully")
  else()
    message(STATUS "ONNX Runtime not found, skipping test_gqa")
  endif()
else()
  message(STATUS "Building test_gqa: DISABLED")
endif()

#=============================================================================
# model_verifier executable for CPU vs GPU output comparison
#=============================================================================
option(BUILD_MODEL_VERIFIER "Build model_verifier executable" ON)

if(BUILD_MODEL_VERIFIER)
  message(STATUS "Building model_verifier: ENABLED")

  if(TARGET onnxruntime::onnxruntime OR ORT_LIB)
    add_executable(model_verifier model_verifier.cpp)

    if(TARGET onnxruntime::onnxruntime)
      target_link_libraries(model_verifier PRIVATE onnxruntime::onnxruntime)
    else()
      target_link_libraries(model_verifier PRIVATE ${ORT_LIB})
      target_include_directories(model_verifier PRIVATE ${ORT_INCLUDE_DIR})
    endif()

    # Also link with ORT from source tree if available
    if(ORT_SOURCE_INCLUDE_DIR)
      target_include_directories(model_verifier PRIVATE "${ORT_SOURCE_INCLUDE_DIR}")
    endif()

    target_compile_features(model_verifier PRIVATE cxx_std_17)

    if(WIN32)
      target_compile_definitions(model_verifier PRIVATE "_CRT_SECURE_NO_WARNINGS")

      # Copy ORT DLLs to output directory
      if(TARGET onnxruntime::onnxruntime)
        add_custom_command(
          TARGET model_verifier POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:onnxruntime::onnxruntime>
            $<TARGET_FILE_DIR:model_verifier>
        )
      endif()
    endif()

    set_target_properties(model_verifier PROPERTIES FOLDER "verification-tests")

    install(TARGETS model_verifier RUNTIME DESTINATION bin)

    message(STATUS "model_verifier configured successfully")
  else()
    message(STATUS "ONNX Runtime not found, skipping model_verifier")
  endif()
else()
  message(STATUS "Building model_verifier: DISABLED")
endif()

#=============================================================================
# ROCm tests (require ROCm libraries)
#=============================================================================

# THEROCK_DIST and CMAKE_PREFIX_PATH are configured in root CMakeLists.txt

find_package(hip QUIET)
find_package(miopen QUIET)
find_package(hipblaslt QUIET)

if(NOT hip_FOUND OR NOT miopen_FOUND OR NOT hipblaslt_FOUND)
  message(STATUS "ROCm libraries not found, skipping ROCm tests")
  return()
endif()

find_package(GTest QUIET)
find_package(glog QUIET)

if(NOT GTest_FOUND OR NOT glog_FOUND)
  message(STATUS "GTest/glog not found, skipping ROCm tests")
  return()
endif()

message(STATUS "Building ROCm tests")

# Timeout test (GPU timeout mechanism) - standalone test for WaitStreamWithTimeout
add_executable(rocm_timeout_test test_timeout.cpp)
target_link_libraries(rocm_timeout_test PRIVATE glog::glog)
target_compile_features(rocm_timeout_test PRIVATE cxx_std_17)
if(WIN32)
  target_include_directories(rocm_timeout_test SYSTEM PRIVATE "${THEROCK_DIST}/include")
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
  set_target_properties(ort_integration_test PROPERTIES FOLDER "rocm-tests")
elseif(ORT_SOURCE_INCLUDE_DIR AND EXISTS "${ORT_BUILD_IMPLIB}")
  # Use ORT from build output (common case when building onnx-hipdnn-ep)
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

# Install ort_integration_test if it was built
if(TARGET ort_integration_test)
  install(TARGETS ort_integration_test RUNTIME DESTINATION bin)
endif()
