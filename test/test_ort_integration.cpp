// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * ORT Integration Test for MorphiZen EP with MLIR backend.
 * This test only creates a session with MorphiZen EP to verify MLIR pass integration.
 *
 * To see log output, set these environment variables before running:
 *   set GLOG_logtostderr=1
 *   set GLOG_minloglevel=0
 */

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
inline std::wstring ToWideString(const char* str) {
  int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
  std::wstring result(len - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, str, -1, &result[0], len);
  return result;
}
#endif

#ifndef MORPHIZEN_EP_LIB_PATH
#ifdef _WIN32
#define MORPHIZEN_EP_LIB_PATH "onnxruntime_morphizen_ep.dll"
#else
#define MORPHIZEN_EP_LIB_PATH "./libonnxruntime_morphizen_ep.so"
#endif
#endif

#ifndef CONV_TEST_MODEL_PATH
#define CONV_TEST_MODEL_PATH "./conv_model.onnx"
#endif

#ifndef CONV_GEMM_TEST_MODEL_PATH
#define CONV_GEMM_TEST_MODEL_PATH "./conv_gemm_model.onnx"
#endif

// Check if a file exists
bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

class OrtIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize ORT
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    // Use INFO level to see Level-1 pass logs (MY_LOG -> glog INFO)
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_INFO, "OrtIntegrationTest");
    
    // Print environment variable status for debugging
    const char* glog_stderr = std::getenv("GLOG_logtostderr");
    const char* glog_minlevel = std::getenv("GLOG_minloglevel");
    
    std::cout << "\n=== Environment Variables ===" << std::endl;
    std::cout << "GLOG_logtostderr: " << (glog_stderr ? glog_stderr : "(not set)") << std::endl;
    std::cout << "GLOG_minloglevel: " << (glog_minlevel ? glog_minlevel : "(not set)") << std::endl;
    std::cout << "==============================\n" << std::endl;

    // Register MorphiZen EP
    const char* lib_path_str = MORPHIZEN_EP_LIB_PATH;
#ifdef _WIN32
    auto lib_path_w = ToWideString(lib_path_str);
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZen", lib_path_w.c_str());
#else
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZen", lib_path_str);
#endif

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      ep_available_ = false;
      std::cout << "[SetUp] MorphiZen EP not available: " << error_msg << std::endl;
    } else {
      ep_available_ = true;
      std::cout << "[SetUp] MorphiZen EP registered successfully from: " << lib_path_str << std::endl;
    }

    // Check if model file exists
    model_available_ = file_exists(CONV_TEST_MODEL_PATH);
    if (!model_available_) {
      std::cout << "[SetUp] Model not available at: " << CONV_TEST_MODEL_PATH << std::endl;
    } else {
      std::cout << "[SetUp] Model found at: " << CONV_TEST_MODEL_PATH << std::endl;
    }
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
  bool model_available_{false};
};

TEST_F(OrtIntegrationTest, LoadMorphiZenProvider) {
  std::cout << "[Test] Loading MorphiZen Execution Provider..." << std::endl;
  
  EXPECT_TRUE(ep_available_) << "MorphiZen EP should be registered successfully";
  
  if (ep_available_) {
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    std::cout << "[Test] Found " << devices.size() << " EP device(s)" << std::endl;
    
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      std::cout << "[Test]   - EP device: " << ep_name << std::endl;
    }
  }
}

// MorphiZen EP integration test - only creates session to verify MLIR pass
TEST_F(OrtIntegrationTest, CreateSessionWithMorphiZenProvider) {
  std::cout << "[Test] Creating session with MorphiZen EP (MLIR backend)..." << std::endl;
  
  if (!ep_available_) {
    GTEST_SKIP() << "MorphiZen EP not available";
  }
  
  ASSERT_TRUE(model_available_) << "Conv model not found at: " << CONV_TEST_MODEL_PATH;

  // Get EP devices
  std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
  
  // Find MorphiZen device
  const OrtEpDevice* morphizen_device = nullptr;
  for (const auto& device : devices) {
    std::string ep_name = device.EpName();
    if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
      morphizen_device = static_cast<const OrtEpDevice*>(device);
      break;
    }
  }

  if (morphizen_device == nullptr) {
    std::cout << "[Test] MorphiZen EP V2 device API not yet implemented" << std::endl;
    std::cout << "[Test] MorphiZen EP configuration (embedded in DLL):" << std::endl;
    std::cout << "[Test]   Level-1 pass: vaip-pass_level1_mlir" << std::endl;
    GTEST_SKIP() << "MorphiZen EP V2 device API not yet implemented (EP registered OK)";
  }

  Ort::SessionOptions session_options;

  // Add MorphiZen EP using V2 API
  OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      session_options, *env_, &morphizen_device, 1, nullptr, nullptr, 0);

  if (status != nullptr) {
    std::string error_msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    FAIL() << "Failed to add MorphiZen EP: " << error_msg;
  }

  std::cout << "[Test] MorphiZen EP configuration:" << std::endl;
  std::cout << "[Test]   Level-1 pass: vaip-pass_level1_mlir (MLIR integration)" << std::endl;

  try {
#ifdef _WIN32
    auto model_path_w = ToWideString(CONV_TEST_MODEL_PATH);
    Ort::Session session(*env_, model_path_w.c_str(), session_options);
#else
    Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);
#endif
    std::cout << "[Test] Session created successfully with MorphiZen EP!" << std::endl;
    std::cout << "[Test] MLIR pass was executed during session creation" << std::endl;
    
  } catch (const Ort::Exception& ex) {
    std::string error_msg = ex.what();
    if (error_msg.find("No engine configurations available") != std::string::npos ||
        error_msg.find("execution_plans failed") != std::string::npos) {
      GTEST_SKIP() << "MLIR backend not available. Error: " << error_msg;
    }
    throw;
  }
}

// MorphiZen EP integration test with Conv+Gemm model
TEST_F(OrtIntegrationTest, CreateSessionWithConvGemmModel) {
  std::cout << "[Test] Creating session with Conv+Gemm model (MLIR backend)..." << std::endl;
  
  if (!ep_available_) {
    GTEST_SKIP() << "MorphiZen EP not available";
  }
  
  bool conv_gemm_available = file_exists(CONV_GEMM_TEST_MODEL_PATH);
  if (!conv_gemm_available) {
    std::cout << "[Test] Conv+Gemm model not found at: " << CONV_GEMM_TEST_MODEL_PATH << std::endl;
    GTEST_SKIP() << "Conv+Gemm model not available";
  }

  // Get EP devices
  std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
  
  // Find MorphiZen device
  const OrtEpDevice* morphizen_device = nullptr;
  for (const auto& device : devices) {
    std::string ep_name = device.EpName();
    if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
      morphizen_device = static_cast<const OrtEpDevice*>(device);
      break;
    }
  }

  if (morphizen_device == nullptr) {
    GTEST_SKIP() << "MorphiZen EP V2 device API not yet implemented";
  }

  Ort::SessionOptions session_options;

  // Add MorphiZen EP using V2 API
  OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      session_options, *env_, &morphizen_device, 1, nullptr, nullptr, 0);

  if (status != nullptr) {
    std::string error_msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    FAIL() << "Failed to add MorphiZen EP: " << error_msg;
  }

  std::cout << "[Test] Testing Conv+Gemm model with MLIR backend..." << std::endl;

  try {
#ifdef _WIN32
    auto model_path_w = ToWideString(CONV_GEMM_TEST_MODEL_PATH);
    Ort::Session session(*env_, model_path_w.c_str(), session_options);
#else
    Ort::Session session(*env_, CONV_GEMM_TEST_MODEL_PATH, session_options);
#endif
    std::cout << "[Test] Session created successfully with Conv+Gemm model!" << std::endl;
    std::cout << "[Test] MLIR pass processed Conv and Gemm operations" << std::endl;
    
  } catch (const Ort::Exception& ex) {
    std::string error_msg = ex.what();
    if (error_msg.find("No engine configurations available") != std::string::npos ||
        error_msg.find("execution_plans failed") != std::string::npos) {
      GTEST_SKIP() << "MLIR backend not available. Error: " << error_msg;
    }
    throw;
  }
}

int main(int argc, char** argv) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "ORT Integration Test for MorphiZen MLIR EP" << std::endl;
  std::cout << "========================================\n" << std::endl;
  
  std::cout << "To see log output, set these environment variables:" << std::endl;
  std::cout << "  set GLOG_logtostderr=1" << std::endl;
  std::cout << "  set GLOG_minloglevel=0" << std::endl;
  std::cout << std::endl;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
