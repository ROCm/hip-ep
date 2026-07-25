/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "test_environment.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "../morphizen-core/src/ep_shared_context_workspace.hpp"

TEST(SharedContextWorkspaceTest, CreateAndGet) {
  auto ep_context_binary_file = CMAKE_CURRENT_BINARY_PATH /
                                std::filesystem::u8path("test_ep_context1.bin");
  auto &workspace =
      morphizen::SharedContextContextWorkspace::create_workspace_or_get(
          ep_context_binary_file);
  EXPECT_EQ(workspace.get_ep_context_binary_file(), ep_context_binary_file);

  // Get the same workspace again

  auto ep_context_binary_file2 =
      CMAKE_CURRENT_BINARY_PATH /
      std::filesystem::u8path("test_ep_context2.bin");
  auto &workspace2 =
      morphizen::SharedContextContextWorkspace::create_workspace_or_get(
          ep_context_binary_file);
  // Get the same workspace again
  auto ep_context_binary_file3 =
      CMAKE_CURRENT_BINARY_PATH / ".." / CMAKE_CURRENT_BINARY_PATH.filename() /
      std::filesystem::u8path("./test_ep_context3.bin");
  auto &workspace3 =
      morphizen::SharedContextContextWorkspace::create_workspace_or_get(
          ep_context_binary_file);
  EXPECT_EQ(&workspace, &workspace3);
}
