##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)

# Find dependencies
find_package(onnxruntime REQUIRED)
find_package(GTest REQUIRED)

# Executable
add_executable(mlir_e2e_test test_e2e_mlir.cpp)

# Require C++17 for std::filesystem
target_compile_features(mlir_e2e_test PRIVATE cxx_std_17)

# Model path (using Git LFS committed model)
set(TWO_LAYER_CONV_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/two_layer_conv.onnx")

# Config file path (absolute path to etc/morphizen_config_mlir.json)
set(MORPHIZEN_CONFIG_MLIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../etc/morphizen_config_mlir.json")

# Get the MorphiZen EP target (set by parent morphizen build system)
if(NOT DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET)
  message(FATAL_ERROR "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined - test requires MorphiZen EP")
endif()

target_compile_definitions(mlir_e2e_test PRIVATE
  TWO_LAYER_CONV_MODEL_PATH="${TWO_LAYER_CONV_MODEL_PATH}"
  MORPHIZEN_EP_LIB_PATH="$<TARGET_FILE:${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}>"
  MORPHIZEN_CONFIG_MLIR_PATH="${MORPHIZEN_CONFIG_MLIR_PATH}"
)

# Pass BUILD_MOCK_RUNTIME flag to test code
if(BUILD_MOCK_RUNTIME)
  target_compile_definitions(mlir_e2e_test PRIVATE BUILD_MOCK_RUNTIME=1)
endif()

target_link_libraries(mlir_e2e_test PRIVATE
  onnxruntime::onnxruntime
  GTest::gtest
  GTest::gtest_main
)

# Copy model to build directory for easier testing
add_custom_command(
  TARGET mlir_e2e_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TWO_LAYER_CONV_MODEL_PATH}"
    "$<TARGET_FILE_DIR:mlir_e2e_test>/two_layer_conv.onnx"
)

# Windows: Copy DLLs to output directory
if(WIN32)
  add_custom_command(
    TARGET mlir_e2e_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:mlir_e2e_test>
  )

  # Copy zlib DLL (required by morphizen EP)
  find_package(ZLIB REQUIRED)
  if(TARGET ZLIB::ZLIB)
    add_custom_command(
      TARGET mlir_e2e_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ZLIB::ZLIB>
        $<TARGET_FILE_DIR:mlir_e2e_test>
    )
  endif()
endif()

# Register test
add_test(NAME MlirE2ETest COMMAND mlir_e2e_test)

set_target_properties(mlir_e2e_test PROPERTIES FOLDER "mlir-tests")
