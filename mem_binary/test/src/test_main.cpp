/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>

int main(int argc, const char* argv[]) {
  testing::InitGoogleTest(&argc, (char**)argv);

  auto ret = RUN_ALL_TESTS();
  return ret;
}
