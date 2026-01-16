// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * ORT Integration Test for VitisAI EP with ROCm backend.
 * This test uses ONNX Runtime with VitisAI Execution Provider to verify
 * the complete pipeline including passes and custom ops (where MY_LOG is used).
 *
 * To see MY_LOG output, set these environment variables before running:
 *   set MORPHIZEN_DEBUG_ROCM=2
 *   set GLOG_logtostderr=1
 *   set GLOG_minloglevel=0
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
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

#ifndef VITISAI_EP_LIB_PATH
#ifdef _WIN32
#define VITISAI_EP_LIB_PATH "onnxruntime_vitisai_ep.dll"
#else
#define VITISAI_EP_LIB_PATH "./libonnxruntime_vitisai_ep.so"
#endif
#endif

#ifndef CONV_TEST_MODEL_PATH
#define CONV_TEST_MODEL_PATH "./conv_model.onnx"
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
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OrtIntegrationTest");
    
    // Print environment variable status for debugging
    const char* debug_level = std::getenv("MORPHIZEN_DEBUG_ROCM");
    const char* glog_stderr = std::getenv("GLOG_logtostderr");
    
    std::cout << "\n=== Environment Variables ===" << std::endl;
    std::cout << "MORPHIZEN_DEBUG_ROCM: " << (debug_level ? debug_level : "(not set)") << std::endl;
    std::cout << "GLOG_logtostderr: " << (glog_stderr ? glog_stderr : "(not set)") << std::endl;
    std::cout << "==============================\n" << std::endl;

    // Register VitisAI EP
    const char* lib_path_str = VITISAI_EP_LIB_PATH;
#ifdef _WIN32
    auto lib_path_w = ToWideString(lib_path_str);
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "VitisAI", lib_path_w.c_str());
#else
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "VitisAI", lib_path_str);
#endif

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      ep_available_ = false;
      std::cout << "[SetUp] VitisAI EP not available: " << error_msg << std::endl;
    } else {
      ep_available_ = true;
      std::cout << "[SetUp] VitisAI EP registered successfully from: " << lib_path_str << std::endl;
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

TEST_F(OrtIntegrationTest, LoadVitisAIProvider) {
  std::cout << "[Test] Loading VitisAI Execution Provider..." << std::endl;
  
  // The EP is registered via RegisterExecutionProviderLibrary
  // Note: VitisAI EP may not yet expose devices via GetEpDevices() (ORT 2.0 API)
  EXPECT_TRUE(ep_available_) << "VitisAI EP should be registered successfully";
  
  if (ep_available_) {
    // Try to get EP devices (ORT 2.0 API)
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    std::cout << "[Test] Found " << devices.size() << " EP device(s)" << std::endl;
    
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      std::cout << "[Test]   - EP device: " << ep_name << std::endl;
    }
    
    // Check for VitisAI device
    bool found_vitisai = false;
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      if (ep_name == "VitisAI" || ep_name == "VitisAIExecutionProvider") {
        found_vitisai = true;
        break;
      }
    }
    
    if (!found_vitisai) {
      std::cout << "[Test] Note: VitisAI EP registered but not exposing V2 devices" << std::endl;
      std::cout << "[Test] This is expected - ORT 2.0 device API may not be implemented yet" << std::endl;
    }
    
    // Test passes if EP was registered (even if V2 API not implemented)
    EXPECT_TRUE(ep_available_);
  }
}

TEST_F(OrtIntegrationTest, CPUProviderInference) {
  std::cout << "[Test] Testing CPU provider inference with conv model..." << std::endl;
  
  ASSERT_TRUE(model_available_) << "Conv model not found at: " << CONV_TEST_MODEL_PATH;
  
  // Model parameters (matches gen_conv_model.py)
  const std::vector<int64_t> input_shape = {1, 3, 8, 8};
  const size_t input_size = 1 * 3 * 8 * 8;

  // Create input data (all 1.0)
  std::vector<float> input_data(input_size, 1.0f);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  
#ifdef _WIN32
  auto model_path_w = ToWideString(CONV_TEST_MODEL_PATH);
  Ort::Session session(*env_, model_path_w.c_str(), session_options);
#else
  Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);
#endif
  
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};

  std::cout << "[Test] Running CPU inference..." << std::endl;
  auto output_tensors = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

  ASSERT_EQ(output_tensors.size(), 1);
  
  auto& output_tensor = output_tensors[0];
  auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  const float* output_data = output_tensor.GetTensorData<float>();
  
  std::cout << "[Test] Output shape: [";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    std::cout << output_shape[i];
    if (i < output_shape.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "[Test] Output[0]: " << output_data[0] << std::endl;

  // Verify output is not all zeros
  bool has_nonzero = false;
  for (size_t i = 0; i < output_size && !has_nonzero; ++i) {
    if (std::abs(output_data[i]) > 1e-6f) {
      has_nonzero = true;
    }
  }

  EXPECT_TRUE(has_nonzero) << "Output should contain non-zero values";
  std::cout << "[Test] CPU inference completed successfully!" << std::endl;
}

// VitisAI EP integration test - tests with Level-1 ROCm pass
TEST_F(OrtIntegrationTest, VitisAIProviderInference) {
  std::cout << "[Test] Testing VitisAI EP with Level-1 ROCm pass..." << std::endl;
  
  if (!ep_available_) {
    GTEST_SKIP() << "VitisAI EP not available";
  }
  
  ASSERT_TRUE(model_available_) << "Conv model not found at: " << CONV_TEST_MODEL_PATH;

  // Model parameters (matches gen_conv_model.py)
  const std::vector<int64_t> input_shape = {1, 3, 8, 8};
  const size_t input_size = 1 * 3 * 8 * 8;

  // Create input data (all 1.0)
  std::vector<float> input_data(input_size, 1.0f);

  // Run with CPU EP first to get reference output
  std::vector<float> cpu_output;
  {
    Ort::SessionOptions session_options;
#ifdef _WIN32
    auto model_path_w = ToWideString(CONV_TEST_MODEL_PATH);
    Ort::Session session(*env_, model_path_w.c_str(), session_options);
#else
    Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);
#endif

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};

    auto output_tensors = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(output_tensors.size(), 1);
    const float* output_data = output_tensors[0].GetTensorData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    cpu_output.assign(output_data, output_data + output_size);
    
    std::cout << "[Test] CPU reference output[0]: " << cpu_output[0] << std::endl;
  }

  // Try to run with VitisAI EP using ORT 2.0 API
  {
    // Get EP devices
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    
    // Find VitisAI device
    const OrtEpDevice* vitisai_device = nullptr;
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      std::cout << "[Test] Found EP device: " << ep_name << std::endl;
      if (ep_name == "VitisAI" || ep_name == "VitisAIExecutionProvider") {
        vitisai_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }

    if (vitisai_device == nullptr) {
      // ORT 2.0 V2 device API not implemented yet in VitisAI EP
      // This is expected - skip with informative message
      std::cout << "[Test] VitisAI EP V2 device API not yet implemented" << std::endl;
      std::cout << "[Test] The EP was registered successfully, but doesn't expose V2 devices" << std::endl;
      std::cout << "[Test] VitisAI EP configuration (embedded in DLL):" << std::endl;
      std::cout << "[Test]   Level-1 pass: vaip-pass_level1_rocm" << std::endl;
      std::cout << "[Test]   Level-2 sub-passes:" << std::endl;
      std::cout << "[Test]     - vaip-pass_level2_rocm_conv" << std::endl;
      std::cout << "[Test]     - vaip-pass_level2_rocm_gemm" << std::endl;
      
      GTEST_SKIP() << "VitisAI EP V2 device API not yet implemented (EP registered OK)";
    }

    Ort::SessionOptions session_options;

    // Add VitisAI EP using V2 API
    OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, *env_, &vitisai_device, 1, nullptr, nullptr, 0);

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      FAIL() << "Failed to add VitisAI EP: " << error_msg;
    }

    std::cout << "[Test] VitisAI EP configuration:" << std::endl;
    std::cout << "[Test]   Level-1 pass: vaip-pass_level1_rocm (ROCm orchestration)" << std::endl;
    std::cout << "[Test]   Level-2 sub-passes:" << std::endl;
    std::cout << "[Test]     - vaip-pass_level2_rocm_conv (Conv pattern matching)" << std::endl;
    std::cout << "[Test]     - vaip-pass_level2_rocm_gemm (Gemm pattern matching)" << std::endl;

    std::cout << "[Test] Creating session with VitisAI EP (ROCm backend)..." << std::endl;

    try {
#ifdef _WIN32
      auto model_path_w = ToWideString(CONV_TEST_MODEL_PATH);
      Ort::Session session(*env_, model_path_w.c_str(), session_options);
#else
      Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);
#endif
      std::cout << "[Test] Session created successfully" << std::endl;

      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

      const char* input_names[] = {"X"};
      const char* output_names[] = {"Y"};

      std::cout << "[Test] Running VitisAI EP inference (MIOpen Conv backend)..." << std::endl;
      auto output_tensors = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);
      std::cout << "[Test] Inference completed" << std::endl;

      ASSERT_EQ(output_tensors.size(), 1);
      auto& output_tensor = output_tensors[0];
      auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
      size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
      const float* output_data = output_tensor.GetTensorData<float>();

      std::vector<float> gpu_output(output_data, output_data + output_size);

      std::cout << "[Test] GPU output shape: [";
      for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "[Test] GPU output[0]: " << gpu_output[0] << std::endl;
      
      // Compare outputs
      ASSERT_EQ(cpu_output.size(), gpu_output.size()) << "Output size mismatch";

      float max_diff = 0.0f;
      for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - gpu_output[i]);
        max_diff = std::max(max_diff, diff);
        EXPECT_NEAR(cpu_output[i], gpu_output[i], 1e-4f)
            << "Mismatch at index " << i << ": CPU=" << cpu_output[i] << ", GPU=" << gpu_output[i];
      }

      std::cout << "[Test] Max difference between CPU and GPU: " << max_diff << std::endl;
      std::cout << "[Test] VitisAI EP inference verified successfully!" << std::endl;
      
    } catch (const Ort::Exception& ex) {
      std::string error_msg = ex.what();
      // Check if the error is due to missing backend
      if (error_msg.find("No engine configurations available") != std::string::npos ||
          error_msg.find("execution_plans failed") != std::string::npos) {
        GTEST_SKIP() << "MIOpen backend not available. This is expected if ROCm is not "
                     << "fully configured. Error: " << error_msg;
      }
      // Re-throw other exceptions
      throw;
    }
  }
}

int main(int argc, char** argv) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "ORT Integration Test for VitisAI ROCm EP" << std::endl;
  std::cout << "========================================\n" << std::endl;
  
  std::cout << "To see MY_LOG output, set these environment variables:" << std::endl;
  std::cout << "  set MORPHIZEN_DEBUG_ROCM=2" << std::endl;
  std::cout << "  set GLOG_logtostderr=1" << std::endl;
  std::cout << "  set GLOG_minloglevel=0" << std::endl;
  std::cout << std::endl;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
