/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/mem_binary.hpp"
#include <gtest/gtest.h>
#include <string>

TEST(MemBinarytest, CheckFileExists) {
  EXPECT_EQ(vaip_core::get_mem_binary_span("sample.bin").has_value(), true);
  EXPECT_EQ(vaip_core::get_mem_binary_span("no_such_file.bin").has_value(),
            false);
  auto content = vaip_core::get_mem_binary_span("sample.bin");
  std::string s(content->data(), content->size());
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  EXPECT_EQ(s, "some magic\n");
}
