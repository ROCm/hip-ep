/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/onnx_schema.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace morphizen {
namespace test {

class OnnxSchemaTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test setup for schema registration
  }

  void TearDown() override {
    // Test cleanup
  }
};

TEST_F(OnnxSchemaTest, TestEPContextOpSchema) {
  // Test that EPContext is registered in com.microsoft domain
  const auto *schema = GetOpSchema("EPContext", "com.microsoft");
  ASSERT_NE(schema, nullptr)
      << "EPContext schema should be registered in com.microsoft domain";

  // Verify schema properties
  EXPECT_EQ(schema->Name(), "EPContext");
  EXPECT_EQ(schema->domain(), "com.microsoft");
  EXPECT_EQ(schema->SinceVersion(), 1);

  LOG(INFO)
      << "EPContext schema successfully registered in com.microsoft domain";
}

} // namespace test
} // namespace morphizen
