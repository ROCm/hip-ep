/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <memory>

namespace morphizen {
class SharedContextContextWorkspace {
private:
  struct PrivateTag {}; // Private tag to prevent direct instantiation
  // Constructor has to be public for std::make_unique
public:
  static SharedContextContextWorkspace &
  create_workspace_or_get(const std::filesystem::path &ep_context_binary_file);

public:
  SharedContextContextWorkspace(
      const PrivateTag &, const std::filesystem::path &ep_context_binary_file);
  SharedContextContextWorkspace(const SharedContextContextWorkspace &) = delete;
  SharedContextContextWorkspace &
  operator=(const SharedContextContextWorkspace &) = delete;
  SharedContextContextWorkspace(SharedContextContextWorkspace &&) = delete;
  void close_workspace();

  // Returns the full path to the EP context binary file
  const std::filesystem::path &get_ep_context_binary_file() const {
    return ep_context_binary_file_;
  }

private:
  const std::filesystem::path ep_context_binary_file_;
};

} // namespace morphizen
