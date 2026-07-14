/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test-coverage-wrapper.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <ort-bridge/ort-global-api.hpp>

namespace morphizen {
namespace test {

TEST_F(TestCoverageWrapperTest, WrapperCreation) {
  // Test that the wrapper was created successfully
  EXPECT_NE(wrapped_api_, nullptr);
  EXPECT_EQ(wrapped_api_->magic, original_api_->magic);
  EXPECT_EQ(wrapped_api_->major, original_api_->major);
  EXPECT_EQ(wrapped_api_->minor, original_api_->minor);
  EXPECT_EQ(wrapped_api_->patch, original_api_->patch);
}

TEST_F(TestCoverageWrapperTest, ApiCallCoverage) {
  // This test demonstrates how to use the coverage wrapper
  // to monitor API calls during testing

  // Example: Call some basic API functions through the wrapper
  auto lib_id = wrapped_api_->get_lib_id();
  auto lib_name = wrapped_api_->get_lib_name();

  EXPECT_FALSE(lib_id.empty());
  EXPECT_FALSE(lib_name.empty());

  LOG(INFO) << "Library ID: " << lib_id;
  LOG(INFO) << "Library Name: " << lib_name;

  // The wrapper will log these calls and count them
  // When TearDown() is called, it will print the statistics
}

TEST_F(TestCoverageWrapperTest, MultipleApiCalls) {
  // Reset statistics before testing
  reset_morphizen_ort_api_call_statistics();

  // Test calling the same API multiple times to verify counting
  for (int i = 0; i < 5; ++i) {
    auto lib_id = wrapped_api_->get_lib_id();
    EXPECT_FALSE(lib_id.empty());
  }

  // Check the statistics programmatically
  auto stats = get_morphizen_ort_api_call_statistics();
  EXPECT_EQ(stats["get_lib_id"], 5);

  LOG(INFO) << "get_lib_id was called " << stats["get_lib_id"] << " times";
}

TEST_F(TestCoverageWrapperTest, StatisticsAccuracy) {
  // Reset statistics
  reset_morphizen_ort_api_call_statistics();

  // Call different APIs a known number of times
  wrapped_api_->get_lib_id();   // 1 call
  wrapped_api_->get_lib_name(); // 1 call
  wrapped_api_->get_lib_id();   // 2nd call

  // Verify statistics
  auto stats = get_morphizen_ort_api_call_statistics();
  EXPECT_EQ(stats["get_lib_id"], 2);
  EXPECT_EQ(stats["get_lib_name"], 1);

  // Verify that uncalled APIs have 0 count
  EXPECT_EQ(stats["model_load"], 0);
}

} // namespace test
} // namespace morphizen
