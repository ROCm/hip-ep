/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./ep_shared_context_workspace.hpp"
#include "./cleanup.hpp"
#include "glog/logging.h"
#include "morphizen/env_config.hpp"
#include <unordered_map>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_EP_CONTEXT_SHARED_WORKSPACE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_EP_CONTEXT_SHARED_WORKSPACE) >= n)

namespace morphizen {
struct path_equal_to {
  bool operator()(const std::filesystem::path& p1,
                  const std::filesystem::path& p2) const {
    return std::filesystem::equivalent(p1, p2);
  }
};
using store_t =
    std::unordered_map<std::filesystem::path,
                       std::unique_ptr<SharedContextContextWorkspace>,
                       std::hash<std::filesystem::path>, path_equal_to>;
static store_t& the_store() {
  static store_t g_store;
  static bool inialized = false;
  if (!inialized) {
    // Register cleanup function to close all workspaces on exit
    add_cleanup_function(std::string(__FILE__) + ":" + std::to_string(__LINE__),
                         []() {
                           for (auto& [_, workspace] : g_store) {
                             workspace->close_workspace();
                           }
                           g_store.clear();
                         });
    inialized = true;
  }
  return g_store;
}

SharedContextContextWorkspace&
SharedContextContextWorkspace::create_workspace_or_get(
    const std::filesystem::path& ep_context_binary_file) {
  auto directory = ep_context_binary_file.has_parent_path()
                       ? ep_context_binary_file.parent_path()
                       : std::filesystem::u8path(".");
  auto filename = ep_context_binary_file.filename().u8string();
  auto& store = the_store();
  auto it = store.find(directory);
  if (it == store.end()) {
    // Create a new workspace if it does not exist
    auto workspace = std::make_unique<SharedContextContextWorkspace>(
        PrivateTag{}, ep_context_binary_file);
    auto ret = workspace.get();
    store[directory] = std::move(workspace);
    MY_LOG(1) << "Creating new workspace for EP context binary file: "
              << ep_context_binary_file;
    return *ret;
  } else {
    // Return the existing workspace
    MY_LOG(1) << "Returning existing workspace for EP context binary file: "
              << ep_context_binary_file
              << " workspace: " << it->second->get_ep_context_binary_file();
    return *it->second.get();
  }
}

SharedContextContextWorkspace::SharedContextContextWorkspace(
    const PrivateTag&, const std::filesystem::path& ep_context_binary_file)
    : ep_context_binary_file_(ep_context_binary_file) {
  // Constructor logic can be added here if needed
}
void SharedContextContextWorkspace::close_workspace() {
  // Cleanup logic can be added here if needed
  MY_LOG(1) << "Closing workspace for EP context binary file: "
            << ep_context_binary_file_;
  auto& store = the_store();
  auto directory = ep_context_binary_file_.has_parent_path()
                       ? ep_context_binary_file_.parent_path()
                       : std::filesystem::u8path(".");

  auto it = store.find(directory);
  if (it != store.end()) {
    store.erase(it);
  } else {
    MY_LOG(1) << "Workspace for EP context binary file not found: is it a bug?"
              << ep_context_binary_file_;
  }
}
} // namespace morphizen
