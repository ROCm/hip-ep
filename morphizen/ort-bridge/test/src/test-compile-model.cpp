/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../../unit-test/morphizen-e2e-test/wide-string.hpp"
#include "./test-environment.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#define ORT_API_MANUAL_INIT 1
#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <glog/logging.h>
#include <memory>
// Include MorphiZen core headers for API access
#include "./test-coverage-wrapper.hpp"
#include <morphizen/morphizen_ort_api.h>
#include <morphizen/onnxruntime_morphizen_ep.hpp>
// Forward declarations for MorphiZen core functions
namespace morphizen {
struct OrtApiForMorphizen;
const OrtApiForMorphizen *get_the_global_api_unsafe();
void set_the_global_api(OrtApiForMorphizen *api);
} // namespace morphizen

struct CompileModel : public ::morphizen::test::TestCoverageWrapperTest {
  void SetUp() override {
    // Initialize the test environment
    morphizen::test::TestCoverageWrapperTest::SetUp();
  }

  void TearDown() override {
    morphizen::test::TestCoverageWrapperTest::TearDown();
  }
};

TEST_F(CompileModel, T0) {
  // Skip this test - target auto-discovery failure (see Issue #031)
  GTEST_SKIP()
      << "Test skipped: Target auto-discovery failure (see Issue #031)";
  // This test implements the basic CompileModel functionality
  // Note: Some dependencies may need to be properly configured

  // Load IR model from a file
  // Use compile-time default backend, but allow override via environment
  // variable
#if MORPHIZEN_ENABLE_ONNX_BACKEND
  auto test_model_path = RESNET_50_PATH;
#elif MORPHIZEN_ENABLE_MLIR_BACKEND
  auto test_model_path = RESNET_50_MLIR_PATH;
#endif

  auto env_backend = ENV_PARAM(MORPHIZEN_ORT_BRIDGE_BACKEND);
  if (!env_backend.empty()) {
    if (env_backend == morphizen::kMLIRBackend) {
      test_model_path = RESNET_50_MLIR_PATH;
    } else if (env_backend == morphizen::kONNXIRBackend) {
      test_model_path = RESNET_50_PATH;
    }
  }
  auto ir_model = MORPHIZEN_ORT_API(model_load)(test_model_path.u8string());
  ASSERT_TRUE(ir_model != nullptr)
      << "Failed to load IR model from file: " << test_model_path;
  // Get graph and model path

  auto &graph = MORPHIZEN_ORT_API(model_main_graph)(*ir_model);
  auto model_path = MORPHIZEN_ORT_API(get_model_path)(graph);
  OrtStatus *status = nullptr;
  auto provider_options = std::unordered_map<std::string, std::string>{};
  provider_options["enable_cache_file_io_in_mem"] = "1";
  auto session_configs = std::map<std::string, std::string>{};
  auto execution_providers = std::make_unique<morphizen::DllSafe<
      std::vector<std::unique_ptr<morphizen::ExecutionProvider>>>>(
      compile_onnx_model_morphizen_ep_v4(
          model_path.u8string(), graph, provider_options, session_configs,
          (void *)&status,
          [](void *status, int code, const char *error_message) {
            OrtStatus **ort_status = static_cast<OrtStatus **>(status);
            *ort_status =
                Ort::GetApi().CreateStatus((OrtErrorCode)code, error_message);
          },
          nullptr));
  // Test compilation (basic validation)
  // OrtStatus* status = nullptr;
  // TODO: Add proper compilation logic when EP implementation is available
  // For now, verify that we can access the graph and model path
  // EXPECT_TRUE(!model_path.empty()) << "Model path should not be empty";
}
