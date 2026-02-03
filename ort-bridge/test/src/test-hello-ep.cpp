/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../../unit-test/morphizen-e2e-test/wide-string.hpp"
#include "./test-environment.hpp"
#define ORT_API_MANUAL_INIT 1
#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include <glog/logging.h>

static void del_ctx_model(const std::filesystem::path& model_path) {
  try {
    std::filesystem::remove(model_path);
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}

struct HelloEpTest : public ::testing::Test {
  void SetUp() override {
    // we must use the reserved name "MorphiZenExecutionProvider" to register
    // the EP library, otherwise the EP will be regarded as a non-cpu EP by the
    // ONNX Runtime, and must implement a kernel for the "MemcopyFromHost" node.
    // see `ProviderIsCpuBased` and `MemcpyTransformer::ApplyImpl` for more
    // details.
    registration_name = "MorphiZenExecutionProvider";
    ort_env =
        std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "HelloEp.Test0");
    LOG(INFO) << "Registering EP library with name: " << registration_name
              << " and library path: " << MORPHIZEN_MORPHIZEN_EP.u8string();
    ASSERT_TRUE(std::filesystem::exists(MORPHIZEN_MORPHIZEN_EP))
        << "EP library does not exist: " << MORPHIZEN_MORPHIZEN_EP.u8string();
    auto status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *ort_env, registration_name.c_str(),
        PathToString<ORTCHAR_T>()(MORPHIZEN_MORPHIZEN_EP).c_str());
    ASSERT_TRUE(status == nullptr)
        << "RegisterExecutionProviderLibrary failed: status = "
        << Ort::GetApi().GetErrorMessage(status);
  }
  void TearDown() override {
    auto status = (Ort::GetApi().UnregisterExecutionProviderLibrary(
        *ort_env, registration_name.c_str()));
    ASSERT_TRUE(status == nullptr)
        << "UnregisterExecutionProviderLibrary failed: status = "
        << Ort::GetApi().GetErrorMessage(status);
  }
  std::string registration_name;
  std::unique_ptr<Ort::Env> ort_env;
};

TEST_F(HelloEpTest, LoadUnloadPluginLibrary) {}

TEST_F(HelloEpTest, CreateSession) {
  // Skip this test - target auto-discovery failure (see Issue #031)
  GTEST_SKIP()
      << "Test skipped: Target auto-discovery failure (see Issue #031)";
  const OrtApi* c_api = &Ort::GetApi();
  Ort::SessionOptions session_options;
  auto ep_devices = ort_env->GetEpDevices();
  if (ep_devices.size() == 0) {
    LOG(INFO) << "No EP devices found for registration name: "
              << registration_name;
    return; // No devices found, skip the test
  }
  std::vector<Ort::ConstEpDevice> selected_devices =
      {}; // Select the first device for testing
  for (const auto& device : ep_devices) {
    LOG(INFO) << "Found EP device: " << device.EpName();
    if (device.EpName() == registration_name) {
      LOG(INFO) << "Selected EP device: " << device.EpName();
      selected_devices.emplace_back(device);
    }
  }
#ifndef _WIN32
  if (selected_devices.empty()) {
    LOG(INFO) << "No devices found for EP: " << registration_name;
    return; // No devices found, skip the test
  }
#endif
  ASSERT_TRUE(!selected_devices.empty())
      << "No devices found for EP: " << registration_name;
  Ort::KeyValuePairs ep_options;
  ep_options.Add("enable_cache_file_io_in_mem", "1");
  auto ctx_model = std::filesystem::u8path("hello_ep_create_session_ctx.onnx");
  if (std::filesystem::exists(ctx_model)) {
    del_ctx_model(ctx_model);
  }
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_file_path",
                                 ctx_model.u8string().c_str());
  session_options.AppendExecutionProvider_V2(*ort_env, selected_devices,
                                             ep_options);
  LOG(INFO) << "Creating session with EP: model_path=" << RESNET_50_PATH;
  Ort::Session session(*ort_env,
                       PathToString<ORTCHAR_T>()(RESNET_50_PATH).c_str(),
                       session_options);
}
