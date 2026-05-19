/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./config.hpp"
#include "./env.hpp"
#include "test_environment.hpp"
#include <algorithm> // std::generate
#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#include <vector>

// Helper function to convert unique_ptr vector to raw pointer vector
// while managing lifetime
const std::vector<const E2ETestConfig*> get_test_configs() {
  static const auto configs = E2ETestConfig::create(E2E_TEST_CONFIG_JSON_PATH);
  std::vector<const E2ETestConfig*> raw_configs;
  raw_configs.reserve(configs.size());

  for (const auto& config : configs) {
    raw_configs.push_back(config.get());
  }

  return raw_configs;
}

class MorphizenE2ETest : public ::testing::TestWithParam<const E2ETestConfig*> {
protected:
  void SetUp() override {}
  void TearDown() override {
    // Cleanup code if needed
  }
};

TEST_P(MorphizenE2ETest, RunE2ETests) {
#ifndef BAZEL_CURRENT_REPOSITORY
  // Skip E2E tests in CMake builds: Issues #032 and #033 are not yet resolved
  // outside the Bazel environment (model path resolution, config loading).
  GTEST_SKIP() << "Test skipped: only enabled for Bazel builds (see Issue "
                  "#032, #033)";
#endif
  const auto* config = GetParam();
  LOG(INFO) << "Running E2E tests with " << config->proto().name()
            << " configurations.";

  auto e2e_test_env =
      std::make_unique<morphizen_e2e_test::E2ETestEnv>(config->proto().env());

  auto e2e_session_options = e2e_test_env->create_e2e_test_session_options();

  for (const auto& session_option : e2e_session_options) {
    auto e2e_sessions = session_option->create_e2e_test_sessions();
    for (auto& session : e2e_sessions) {
      session->run();
    }
  }

  LOG(INFO) << "E2E tests completed for config: " << config->proto().name();
}

INSTANTIATE_TEST_SUITE_P(
    MorphizenE2ETestSuite, MorphizenE2ETest,
    ::testing::ValuesIn(get_test_configs()),
    [](const testing::TestParamInfo<const E2ETestConfig*>& info) {
      return info.param->proto().name();
    });
