/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <morphizen-utils/morphizen-utils.hpp>
#include <morphizen/morphizen.hpp>
#include <morphizen/morphizen_ort_api.h>
#include <string>

// Determine default backend based on compile-time configuration
#if MORPHIZEN_ENABLE_ONNX_BACKEND
#  define MORPHIZEN_DEFAULT_BACKEND morphizen::kONNXIRBackend
#elif MORPHIZEN_ENABLE_MLIR_BACKEND
#  define MORPHIZEN_DEFAULT_BACKEND morphizen::kMLIRBackend
#else
#  error                                                                       \
      "At least one backend must be enabled: MORPHIZEN_ENABLE_ONNX_BACKEND or MORPHIZEN_ENABLE_MLIR_BACKEND"
#endif

DEF_ENV_PARAM_2(
    MORPHIZEN_ORT_BRIDGE_BACKEND,
    MORPHIZEN_DEFAULT_BACKEND, // default depends on which backend is enabled
    std::string)

namespace morphizen {
// defined in onnx-ir-imp/src/morphizen-ort-api.cpp
const morphizen::OrtApiForMorphizen*
get_global_morphizen_ort_api(const char* ir_backend_name);

namespace test {

/**
 * @brief Create a test coverage wrapper for OrtApiForMorphizen
 *
 * This function creates a wrapper implementation that logs API calls
 * and delegates to the original API for test coverage purposes.
 *
 * @param original_api The original OrtApiForMorphizen instance to wrap
 * @return OrtApiForMorphizen* Wrapped API instance for testing
 */
morphizen::OrtApiForMorphizen* get_morphizen_ort_api_for_coverage_test(
    morphizen::OrtApiForMorphizen* original_api);

/**
 * @brief Delete the test coverage wrapper
 *
 * @param wrapped_api The wrapped API instance to delete
 */
void delete_morphizen_ort_api_coverage_test(
    morphizen::OrtApiForMorphizen* wrapped_api);

/**
 * @brief Get current API call statistics
 *
 * @return std::map<std::string, size_t> Map of API function names to call
 * counts
 */
std::map<std::string, size_t> get_morphizen_ort_api_call_statistics();

/**
 * @brief Reset API call statistics
 */
void reset_morphizen_ort_api_call_statistics();

class TestCoverageWrapperTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Enable verbose logging for API calls
    FLAGS_v = 3;
    auto env_backend = ENV_PARAM(MORPHIZEN_ORT_BRIDGE_BACKEND);
    if (env_backend.empty()) {
      backend_ = MORPHIZEN_DEFAULT_BACKEND;
    } else {
      backend_ = env_backend;
    }
    // Get the original API
    original_api_ = const_cast<morphizen::OrtApiForMorphizen*>(
        morphizen::get_global_morphizen_ort_api(backend_.c_str()));
    // Create the coverage wrapper
    wrapped_api_ = get_morphizen_ort_api_for_coverage_test(original_api_);
    ASSERT_NE(wrapped_api_, nullptr);
    morphizen::set_the_global_api(wrapped_api_);
  }

  void TearDown() override {
    // Clean up the wrapper and print statistics
    if (wrapped_api_) {
      delete_morphizen_ort_api_coverage_test(wrapped_api_);
      wrapped_api_ = nullptr;
    }
  }

  std::string backend_;
  morphizen::OrtApiForMorphizen* original_api_ = nullptr;
  morphizen::OrtApiForMorphizen* wrapped_api_ = nullptr;
};
} // namespace test
} // namespace morphizen
