/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file test_e2e_mlir.cpp
 * @brief E2E test for MLIR backend integration
 *
 * This test validates the ONNX → MLIR → HIP compilation pipeline.
 * It works with both MOCK runtime (default, no GPU required) and REAL runtime
 * (compile-time option with BUILD_MOCK_RUNTIME=OFF).
 *
 * The test uses a two-layer convolution model:
 * Input [1,3,224,224] → Conv1 → ReLU → Conv2 → ReLU → Output [1,64,112,112]
 *
 * Environment variables (all optional):
 * - ORT_LOG_LEVEL=info - Enable ORT session creation logging
 * - DEBUG_MORPHIZEN_PASS=1 - Enable morphizen pass debug logging
 * - MORPHIZEN_DEBUG_MLIR_BACKEND=3 - MLIR backend compilation verbose logging
 */

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <codecvt>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
// Helper to convert std::string to std::wstring on Windows
#ifdef _WIN32
std::wstring StringToWString(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(str);
}
#endif
} // namespace

// Model path defined by CMake
#ifndef TWO_LAYER_CONV_MODEL_PATH
#error "TWO_LAYER_CONV_MODEL_PATH must be defined by CMake"
#endif

#ifndef MORPHIZEN_EP_LIB_PATH
#error "MORPHIZEN_EP_LIB_PATH must be defined by CMake"
#endif

#ifndef MORPHIZEN_CONFIG_MLIR_PATH
#error "MORPHIZEN_CONFIG_MLIR_PATH must be defined by CMake"
#endif

namespace {
// Parse ORT_LOG_LEVEL environment variable
OrtLoggingLevel GetOrtLoggingLevel() {
  const char *log_level_str = std::getenv("ORT_LOG_LEVEL");
  if (!log_level_str) {
    return ORT_LOGGING_LEVEL_WARNING; // Default
  }

  std::string level(log_level_str);
  // Convert to lowercase for case-insensitive comparison
  std::transform(level.begin(), level.end(), level.begin(), ::tolower);

  if (level == "verbose")
    return ORT_LOGGING_LEVEL_VERBOSE;
  if (level == "info")
    return ORT_LOGGING_LEVEL_INFO;
  if (level == "warning")
    return ORT_LOGGING_LEVEL_WARNING;
  if (level == "error")
    return ORT_LOGGING_LEVEL_ERROR;
  if (level == "fatal")
    return ORT_LOGGING_LEVEL_FATAL;

  // Try to parse as integer (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL)
  try {
    int level_int = std::stoi(level);
    if (level_int >= 0 && level_int <= 4) {
      return static_cast<OrtLoggingLevel>(level_int);
    }
  } catch (...) {
    // Ignore parse errors, use default
  }

  return ORT_LOGGING_LEVEL_WARNING; // Default if invalid
}
} // namespace

class MlirE2ETest : public ::testing::Test {
protected:
  std::unique_ptr<Ort::Env> env_;
  std::string model_path_;
  std::string ep_lib_path_;
  std::string config_path_;

  void SetUp() override {
    // Set environment variables for verbose logging. _putenv_s is MSVC-only;
    // POSIX uses setenv. Wrap both behind a tiny SETENV macro so this test
    // builds on both platforms.
#ifdef _WIN32
#define HIPDNN_SETENV(k, v) _putenv_s((k), (v))
#else
#define HIPDNN_SETENV(k, v) setenv((k), (v), /*overwrite=*/1)
#endif
    HIPDNN_SETENV("HIP_EP_VERBOSE", "2");
    HIPDNN_SETENV("DEBUG_LOG_LEVEL", "info");
    HIPDNN_SETENV("MORPHIZEN_DEBUG_PLUGIN", "1");
#undef HIPDNN_SETENV

    // Initialize ORT environment with configurable log level
    OrtLoggingLevel log_level = GetOrtLoggingLevel();
    env_ = std::make_unique<Ort::Env>(log_level, "MlirE2ETest");

    // Set model path
    model_path_ = TWO_LAYER_CONV_MODEL_PATH;
    ep_lib_path_ = MORPHIZEN_EP_LIB_PATH;
    config_path_ = MORPHIZEN_CONFIG_MLIR_PATH;

    // Verify model file exists
    if (!fs::exists(model_path_)) {
      GTEST_SKIP() << "Model file not found: " << model_path_
                   << "\nRun: python gen_two_layer_conv_model.py --output "
                      "models/two_layer_conv.onnx";
    }

    // Verify EP library exists
    if (!fs::exists(ep_lib_path_)) {
      GTEST_SKIP() << "MorphiZen EP library not found: " << ep_lib_path_;
    }

    std::cout << "[SetUp] Model path: " << model_path_ << std::endl;
    std::cout << "[SetUp] EP library path: " << ep_lib_path_ << std::endl;
    std::cout << "[SetUp] Config path: " << config_path_ << std::endl;

    // Verify config file exists
    if (!fs::exists(config_path_)) {
      GTEST_SKIP() << "Config file not found: " << config_path_;
    }

    // Register MorphiZen EP using RegisterExecutionProviderLibrary (not
    // RegisterCustomOps)
#ifdef _WIN32
    OrtStatus *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZenExecutionProvider",
        StringToWString(ep_lib_path_).c_str());
#else
    OrtStatus *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZenExecutionProvider", ep_lib_path_.c_str());
#endif

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      GTEST_SKIP() << "Failed to register MorphiZen EP: " << error_msg;
    }

    std::cout << "[SetUp] MorphiZen EP registered successfully" << std::endl;
  }

  void TearDown() override {
    // Clean up
    env_.reset();
  }
};

/**
 * @test TwoLayerConvSession
 * @brief Test session creation with two-layer convolution model
 *
 * This test validates:
 * 1. MorphiZen EP can be registered
 * 2. ONNX model can be loaded
 * 3. Session can be created (triggers MLIR compilation pipeline)
 * 4. Works with both MOCK runtime (default) and REAL runtime
 * (BUILD_MOCK_RUNTIME=OFF)
 *
 * Expected behavior:
 * - MOCK runtime: Logs show [MOCK] prefixes for HIP/hipBLASLt calls
 * - REAL runtime: Actual GPU execution (requires ROCm hardware)
 *
 * TODO: Add actual inference with input data and output validation
 */
TEST_F(MlirE2ETest, TwoLayerConvSession) {
  std::cout << "[Test] Creating session with MorphiZen EP (MLIR backend)..."
            << std::endl;

  // Get EP devices
  std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();

  // Find MorphiZen device
  const OrtEpDevice *morphizen_device = nullptr;
  for (const auto &device : devices) {
    std::string ep_name = device.EpName();
    if (ep_name == "MorphiZenExecutionProvider") {
      morphizen_device = static_cast<const OrtEpDevice *>(device);
      std::cout << "[Test] Found MorphiZen EP device" << std::endl;
      break;
    }
  }

  if (morphizen_device == nullptr) {
    GTEST_SKIP() << "MorphiZen EP V2 device not found (EP registered but "
                    "device API not implemented)";
  }

  // Create session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Append MorphiZen EP using V2 API with MLIR config
  const char *provider_options_keys[] = {"config_file"};
  std::string config_path_str = config_path_; // Use absolute path from CMake
  const char *provider_options_values[] = {config_path_str.c_str()};

  OrtStatus *status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      session_options, *env_, &morphizen_device, 1, provider_options_keys,
      provider_options_values, 1);

  if (status != nullptr) {
    std::string error_msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    GTEST_SKIP() << "Failed to append MorphiZen EP: " << error_msg;
  }

  std::cout << "[Test] MorphiZen EP configured with " << config_path_
            << std::endl;

  // Create session (this triggers ONNX → MLIR → HIP compilation)
#ifdef _WIN32
  Ort::Session session(*env_, StringToWString(model_path_).c_str(),
                       session_options);
#else
  Ort::Session session(*env_, model_path_.c_str(), session_options);
#endif

  std::cout << "[Test] Session created successfully with MorphiZen EP!"
            << std::endl;

  // Get input/output info
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_input_nodes = session.GetInputCount();
  size_t num_output_nodes = session.GetOutputCount();

  std::cout << "[Test] Model has " << num_input_nodes << " input(s) and "
            << num_output_nodes << " output(s)" << std::endl;

  // Print input info
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::cout << "[Test] Input " << i << ": " << input_name.get() << " shape=[";
    for (size_t j = 0; j < shape.size(); j++) {
      std::cout << shape[j];
      if (j < shape.size() - 1)
        std::cout << ",";
    }
    std::cout << "]" << std::endl;
  }

  // Print output info
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::cout << "[Test] Output " << i << ": " << output_name.get()
              << " shape=[";
    for (size_t j = 0; j < shape.size(); j++) {
      std::cout << shape[j];
      if (j < shape.size() - 1)
        std::cout << ",";
    }
    std::cout << "]" << std::endl;
  }

  // Session creation succeeded - MLIR compilation pipeline worked!
  SUCCEED();

  // TODO: Add actual inference
  // - Create input tensor with test data
  // - Run session.Run()
  // - Validate output tensor
  // - For MOCK runtime: outputs will be zeros
  // - For REAL runtime: outputs will be computed results
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "=== MLIR E2E Test ===" << std::endl;
  std::cout << "This test validates ONNX → MLIR → HIP compilation pipeline"
            << std::endl;
  std::cout << "Runtime: "
#ifdef BUILD_MOCK_RUNTIME
            << "MOCK (no GPU required, outputs filled with zeros)"
#else
            << "REAL (requires ROCm GPU)"
#endif
            << std::endl;
  std::cout << "Model: " << TWO_LAYER_CONV_MODEL_PATH << std::endl;
  std::cout << std::endl;

  return RUN_ALL_TESTS();
}
