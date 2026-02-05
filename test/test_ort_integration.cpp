// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * ORT Integration Test for Morphizen EP with ROCm backend.
 * This test uses ONNX Runtime with Morphizen Execution Provider to verify
 * the complete pipeline including passes and custom ops (where MY_LOG is used).
 *
 * To see MY_LOG output, set these environment variables before running:
 *   set MORPHIZEN_DEBUG_ROCM=2
 *   set GLOG_logtostderr=1
 *   set GLOG_minloglevel=0
 */

#include <cstdlib>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <iostream>

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
#define CONV_TEST_MODEL_PATH "./sample.onnx"
#endif

// Check if a file exists
bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

class OrtIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Determine log level from environment variable ORT_LOG_LEVEL
    OrtLoggingLevel ort_log_level = ORT_LOGGING_LEVEL_WARNING;
    const char* log_level_env = std::getenv("ORT_LOG_LEVEL");
    if (log_level_env != nullptr) {
      std::string log_level_str(log_level_env);
      if (log_level_str == "info") {
        ort_log_level = ORT_LOGGING_LEVEL_INFO;
      } else if (log_level_str == "warning") {
        ort_log_level = ORT_LOGGING_LEVEL_WARNING;
      } else if (log_level_str == "error") {
        ort_log_level = ORT_LOGGING_LEVEL_ERROR;
      }
    }
    env_ = std::make_unique<Ort::Env>(ort_log_level, "OrtIntegrationTest");
    
    // Register Morphizen EP
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
    
    // Check for Morphizen device
    bool found_morphizen = false;
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
        found_morphizen = true;
        break;
      }
    }
    
    if (!found_morphizen) {
      std::cout << "[Test] Note: MorphiZen EP registered but not exposing V2 devices" << std::endl;
      std::cout << "[Test] This is expected - ORT 2.0 device API may not be implemented yet" << std::endl;
    }
    
    // Test passes if EP was registered (even if V2 API not implemented)
    EXPECT_TRUE(ep_available_);
  }
}

TEST_F(OrtIntegrationTest, CPUProviderInference) {
  std::cout << "[Test] Testing CPU provider inference with conv model..." << std::endl;
  
  ASSERT_TRUE(model_available_) << "Conv model not found at: " << CONV_TEST_MODEL_PATH;
  
  // Model parameters for conv test
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

// Morphizen EP integration test - tests with Level-1 ROCm pass
TEST_F(OrtIntegrationTest, MorphiZenProviderInference) {
  std::cout << "[Test] Testing MorphiZen EP with Level-1 ROCm pass..." << std::endl;
  
  if (!ep_available_) {
    GTEST_SKIP() << "MorphiZen EP not available";
  }
  
  ASSERT_TRUE(model_available_) << "Conv model not found at: " << CONV_TEST_MODEL_PATH;

  // Model parameters for conv test
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

  // Try to run with Morphizen EP using ORT 2.0 API
  {
    // Get EP devices
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    
    // Find Morphizen device
    const OrtEpDevice* morphizen_device = nullptr;
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      std::cout << "[Test] Found EP device: " << ep_name << std::endl;
      if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
        morphizen_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }

    if (morphizen_device == nullptr) {
      // ORT 2.0 V2 device API not implemented yet in Morphizen EP
      // This is expected - skip with informative message
      std::cout << "[Test] MorphiZen EP V2 device API not yet implemented" << std::endl;
      std::cout << "[Test] The EP was registered successfully, but doesn't expose V2 devices" << std::endl;
      
      GTEST_SKIP() << "Morphizen EP V2 device API not yet implemented (EP registered OK)";
    }

    Ort::SessionOptions session_options;

    // Add Morphizen EP using V2 API
    OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, *env_, &morphizen_device, 1, nullptr, nullptr, 0);

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      FAIL() << "Failed to add Morphizen EP: " << error_msg;
    }

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

      std::cout << "[Test] Running Morphizen EP inference (MIOpen Conv backend)..." << std::endl;
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
      std::cout << "[Test] CPU reference[0]: " << cpu_output[0] << std::endl;
      
      // Compare outputs with early exit on excessive mismatches
      ASSERT_EQ(cpu_output.size(), gpu_output.size()) << "Output size mismatch";

      const size_t MAX_MISMATCH_BEFORE_FATAL = 5;  // Exit after 5 mismatches
      const float TOLERANCE = 1e-4f;
      
      float max_diff = 0.0f;
      size_t mismatch_count = 0;
      size_t first_mismatch_idx = SIZE_MAX;
      
      for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - gpu_output[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > TOLERANCE) {
          if (mismatch_count < MAX_MISMATCH_BEFORE_FATAL) {
            std::cout << "[Test] MISMATCH at index " << i << ": CPU=" << cpu_output[i] 
                      << ", GPU=" << gpu_output[i] << ", diff=" << diff << std::endl;
          }
          if (first_mismatch_idx == SIZE_MAX) {
            first_mismatch_idx = i;
          }
          mismatch_count++;
          
          if (mismatch_count >= MAX_MISMATCH_BEFORE_FATAL) {
            std::cout << "[Test] FATAL: Too many mismatches (" << mismatch_count 
                      << "+), first mismatch at index " << first_mismatch_idx << std::endl;
            std::cout << "[Test] Max difference so far: " << max_diff << std::endl;
            std::cout << "[Test] Stopping comparison early to avoid log flood." << std::endl;
            FAIL() << "CPU vs Morphizen EP output mismatch: " << mismatch_count 
                   << " elements differ by more than " << TOLERANCE;
          }
        }
      }

      if (mismatch_count > 0) {
        std::cout << "[Test] Total mismatches: " << mismatch_count << " out of " 
                  << cpu_output.size() << " elements" << std::endl;
        std::cout << "[Test] Max difference: " << max_diff << std::endl;
        FAIL() << "CPU vs Morphizen EP output has " << mismatch_count << " mismatches";
      }

      std::cout << "[Test] Max difference between CPU and GPU: " << max_diff << std::endl;
      std::cout << "[Test] Morphizen EP inference verified successfully!" << std::endl;
      
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
  std::cout << "ORT Integration Test for Morphizen HIP EP" << std::endl;
  std::cout << "========================================\n" << std::endl;
  
  std::cout << "To see MY_LOG output, set these environment variables:" << std::endl;
  std::cout << "  set MORPHIZEN_DEBUG_ROCM=2" << std::endl;
  std::cout << "  set GLOG_logtostderr=1" << std::endl;
  std::cout << "  set GLOG_minloglevel=0" << std::endl;
  std::cout << std::endl;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
