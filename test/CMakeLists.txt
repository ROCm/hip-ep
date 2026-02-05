##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building MLIR tests")

# GTest is provided by MorphiZen's FetchContent with OVERRIDE_FIND_PACKAGE
find_package(GTest REQUIRED)

# ORT Integration test (for MorphiZen EP testing with MLIR pass)
# This test links against ONNX Runtime and can trigger the MLIR pass

# Try to find ONNX Runtime using find_package
find_package(onnxruntime QUIET)

if(TARGET onnxruntime::onnxruntime)
  message(STATUS "Building ORT integration test (using onnxruntime::onnxruntime target)")

  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    onnxruntime::onnxruntime
    GTest::gtest
    GTest::gtest_main
  )
  target_compile_features(ort_integration_test PRIVATE cxx_std_17)
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
else()
  message(STATUS "ONNX Runtime not found, skipping ORT integration test")
  message(STATUS "  Hint: Ensure onnxruntime is installed and CMAKE_PREFIX_PATH includes the installation directory")
endif()
