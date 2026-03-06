/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file test_e2e_hip.cpp
 * @brief E2E test for the HIP compiler backend through the MorphiZen EP.
 *
 * Validates the full pipeline:
 *   ONNX model -> MorphiZen EP -> HIP compiler -> DLL -> inference
 *
 * Uses the attention model (2-head self-attention):
 *   Input [2,64,64] -> Q/K/V projections -> scaled dot-product attention
 *   -> Output [2,64,64]
 *
 * Environment variables (all optional):
 * - ORT_LOG_LEVEL=info        Enable ORT session creation logging
 * - HIP_COMPILER_DEBUG=3      HIP compiler verbose logging
 * - DEBUG_MORPHIZEN_PASS=1    MorphiZen pass debug logging
 */

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

namespace fs = std::filesystem;

#ifndef ATTENTION_MODEL_PATH
#error "ATTENTION_MODEL_PATH must be defined by CMake"
#endif

#ifndef MORPHIZEN_EP_LIB_PATH
#error "MORPHIZEN_EP_LIB_PATH must be defined by CMake"
#endif

#ifndef MORPHIZEN_CONFIG_HIP_PATH
#error "MORPHIZEN_CONFIG_HIP_PATH must be defined by CMake"
#endif

namespace {

#ifdef _WIN32
std::wstring StringToWString(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(str);
}
#endif

OrtLoggingLevel GetOrtLoggingLevel() {
  const char *s = std::getenv("ORT_LOG_LEVEL");
  if (!s)
    return ORT_LOGGING_LEVEL_WARNING;
  std::string level(s);
  std::transform(level.begin(), level.end(), level.begin(), ::tolower);
  if (level == "verbose")
    return ORT_LOGGING_LEVEL_VERBOSE;
  if (level == "info")
    return ORT_LOGGING_LEVEL_INFO;
  if (level == "warning")
    return ORT_LOGGING_LEVEL_WARNING;
  if (level == "error")
    return ORT_LOGGING_LEVEL_ERROR;
  return ORT_LOGGING_LEVEL_WARNING;
}

// Run the model on CPU EP to produce reference output
std::vector<float> runCpuReference(const std::string &modelPath,
                                   const std::vector<float> &inputData,
                                   const std::vector<int64_t> &inputShape) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpu_ref");
  Ort::SessionOptions opts;

#ifdef _WIN32
  Ort::Session session(env, StringToWString(modelPath).c_str(), opts);
#else
  Ort::Session session(env, modelPath.c_str(), opts);
#endif

  auto inName =
      session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto outName =
      session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto inTensor = Ort::Value::CreateTensor<float>(
      memInfo, const_cast<float *>(inputData.data()), inputData.size(),
      inputShape.data(), inputShape.size());

  const char *inNames[] = {inName.get()};
  const char *outNames[] = {outName.get()};
  auto outputs = session.Run(Ort::RunOptions{}, inNames, &inTensor, 1,
                             outNames, 1);

  const float *data = outputs[0].GetTensorData<float>();
  auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
  size_t total = 1;
  for (auto d : outInfo.GetShape())
    total *= d;
  return std::vector<float>(data, data + total);
}

} // namespace

class HipE2ETest : public ::testing::Test {
protected:
  std::unique_ptr<Ort::Env> env_;
  std::string model_path_;
  std::string ep_lib_path_;
  std::string config_path_;

  void SetUp() override {
#ifdef _WIN32
    _putenv_s("XLNX_ONNX_EP_VERBOSE", "2");
    _putenv_s("MORPHIZEN_DEBUG_PLUGIN", "1");
#else
    setenv("XLNX_ONNX_EP_VERBOSE", "2", 1);
    setenv("MORPHIZEN_DEBUG_PLUGIN", "1", 1);
#endif

    env_ = std::make_unique<Ort::Env>(GetOrtLoggingLevel(), "HipE2ETest");

    model_path_ = ATTENTION_MODEL_PATH;
    ep_lib_path_ = MORPHIZEN_EP_LIB_PATH;
    config_path_ = MORPHIZEN_CONFIG_HIP_PATH;

    if (!fs::exists(model_path_))
      GTEST_SKIP() << "Model not found: " << model_path_;
    if (!fs::exists(ep_lib_path_))
      GTEST_SKIP() << "EP library not found: " << ep_lib_path_;
    if (!fs::exists(config_path_))
      GTEST_SKIP() << "Config not found: " << config_path_;

    std::cout << "[SetUp] Model:  " << model_path_ << "\n";
    std::cout << "[SetUp] EP lib: " << ep_lib_path_ << "\n";
    std::cout << "[SetUp] Config: " << config_path_ << "\n";

#ifdef _WIN32
    OrtStatus *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZenExecutionProvider",
        StringToWString(ep_lib_path_).c_str());
#else
    OrtStatus *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "MorphiZenExecutionProvider", ep_lib_path_.c_str());
#endif

    if (status) {
      std::string msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      GTEST_SKIP() << "Failed to register EP: " << msg;
    }
    std::cout << "[SetUp] MorphiZen EP registered\n";
  }

  void TearDown() override { env_.reset(); }
};

TEST_F(HipE2ETest, AttentionSessionCreation) {
  std::cout << "[Test] Creating session with HIP backend config...\n";

  auto devices = env_->GetEpDevices();
  const OrtEpDevice *device = nullptr;
  for (const auto &d : devices) {
    if (std::string(d.EpName()) == "MorphiZenExecutionProvider") {
      device = static_cast<const OrtEpDevice *>(d);
      break;
    }
  }
  if (!device)
    GTEST_SKIP() << "MorphiZen device not found";

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

  const char *keys[] = {"config_file"};
  std::string cfgStr = config_path_;
  const char *vals[] = {cfgStr.c_str()};

  OrtStatus *status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      opts, *env_, &device, 1, keys, vals, 1);
  if (status) {
    std::string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    GTEST_SKIP() << "Failed to append EP: " << msg;
  }

#ifdef _WIN32
  Ort::Session session(*env_, StringToWString(model_path_).c_str(), opts);
#else
  Ort::Session session(*env_, model_path_.c_str(), opts);
#endif

  std::cout << "[Test] Session created (HIP compilation pipeline succeeded)\n";

  size_t numIn = session.GetInputCount();
  size_t numOut = session.GetOutputCount();
  std::cout << "[Test] Inputs: " << numIn << ", Outputs: " << numOut << "\n";

  EXPECT_GE(numIn, 1u);
  EXPECT_GE(numOut, 1u);
}

TEST_F(HipE2ETest, AttentionInference) {
  auto devices = env_->GetEpDevices();
  const OrtEpDevice *device = nullptr;
  for (const auto &d : devices) {
    if (std::string(d.EpName()) == "MorphiZenExecutionProvider") {
      device = static_cast<const OrtEpDevice *>(d);
      break;
    }
  }
  if (!device)
    GTEST_SKIP() << "MorphiZen device not found";

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

  const char *keys[] = {"config_file"};
  std::string cfgStr = config_path_;
  const char *vals[] = {cfgStr.c_str()};

  OrtStatus *status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      opts, *env_, &device, 1, keys, vals, 1);
  if (status) {
    std::string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    GTEST_SKIP() << "Failed to append EP: " << msg;
  }

#ifdef _WIN32
  Ort::Session session(*env_, StringToWString(model_path_).c_str(), opts);
#else
  Ort::Session session(*env_, model_path_.c_str(), opts);
#endif

  // Prepare input: [2, 64, 64]
  const int64_t B = 2, S = 64, D = 64;
  const int64_t total = B * S * D;
  std::vector<int64_t> shape = {B, S, D};

  std::vector<float> inputData(total);
  srand(42);
  for (auto &v : inputData)
    v = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

  auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto inTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputData.data(), total, shape.data(), shape.size());

  Ort::AllocatorWithDefaultOptions alloc;
  auto inName = session.GetInputNameAllocated(0, alloc);
  auto outName = session.GetOutputNameAllocated(0, alloc);
  const char *inNames[] = {inName.get()};
  const char *outNames[] = {outName.get()};

  std::cout << "[Test] Running inference via HIP EP...\n";
  auto outputs =
      session.Run(Ort::RunOptions{}, inNames, &inTensor, 1, outNames, 1);

  const float *gpuData = outputs[0].GetTensorData<float>();
  auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
  auto outShape = outInfo.GetShape();
  size_t outTotal = 1;
  for (auto d : outShape)
    outTotal *= d;

  std::cout << "[Test] GPU output shape: [";
  for (size_t i = 0; i < outShape.size(); ++i) {
    std::cout << outShape[i];
    if (i < outShape.size() - 1)
      std::cout << ",";
  }
  std::cout << "] (" << outTotal << " elements)\n";

  // Run CPU reference
  std::cout << "[Test] Running CPU reference...\n";
  auto cpuRef = runCpuReference(model_path_, inputData, shape);

  ASSERT_EQ(outTotal, cpuRef.size());

  // Compare
  float maxDiff = 0;
  int64_t worstIdx = 0;
  for (size_t i = 0; i < outTotal; ++i) {
    float d = std::fabs(gpuData[i] - cpuRef[i]);
    if (d > maxDiff) {
      maxDiff = d;
      worstIdx = i;
    }
  }

  std::cout << "[Test] Max abs diff: " << maxDiff << " (at index " << worstIdx
            << ": gpu=" << gpuData[worstIdx] << " cpu=" << cpuRef[worstIdx]
            << ")\n";

  EXPECT_LT(maxDiff, 1e-2f) << "GPU vs CPU mismatch exceeds tolerance";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "=== HIP Compiler E2E Test ===\n";
  std::cout << "Pipeline: ONNX model -> MorphiZen EP -> HIP compiler -> DLL "
               "-> inference\n";
  std::cout << "Model: " << ATTENTION_MODEL_PATH << "\n\n";

  return RUN_ALL_TESTS();
}
