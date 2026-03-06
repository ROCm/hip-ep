##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)

find_package(onnxruntime REQUIRED)
find_package(GTest REQUIRED)

add_executable(test-e2e-hip test_e2e_hip.cpp)

target_compile_features(test-e2e-hip PRIVATE cxx_std_17)

# Model: attention.onnx from examples/
set(ATTENTION_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/attention.onnx")

# Config: etc/morphizen_config_hip.json
set(MORPHIZEN_CONFIG_HIP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../etc/morphizen_config_hip.json")

if(NOT DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET)
  message(FATAL_ERROR "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined")
endif()

target_compile_definitions(test-e2e-hip PRIVATE
  ATTENTION_MODEL_PATH="${ATTENTION_MODEL_PATH}"
  MORPHIZEN_EP_LIB_PATH="$<TARGET_FILE:${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}>"
  MORPHIZEN_CONFIG_HIP_PATH="${MORPHIZEN_CONFIG_HIP_PATH}"
)

target_link_libraries(test-e2e-hip PRIVATE
  onnxruntime::onnxruntime
  GTest::gtest
  GTest::gtest_main
)

# Copy model to build dir
if(EXISTS "${ATTENTION_MODEL_PATH}")
  add_custom_command(
    TARGET test-e2e-hip POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ATTENTION_MODEL_PATH}"
      "$<TARGET_FILE_DIR:test-e2e-hip>/attention.onnx"
  )
endif()

# Windows: copy onnxruntime.dll
if(WIN32)
  add_custom_command(
    TARGET test-e2e-hip POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:test-e2e-hip>
  )
endif()

add_test(NAME HipE2ETest COMMAND test-e2e-hip)
set_target_properties(test-e2e-hip PROPERTIES FOLDER "hip-tests")
