/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen-utils/cleanup.hpp"
#include "morphizen-foundation/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <utility>
#include <vector>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE) >= n)

namespace morphizen {
std::vector<std::pair<std::string, std::function<void()>>>&
get_cleanup_registry() {
  static std::vector<std::pair<std::string, std::function<void()>>> g_at_exits;
  return g_at_exits;
}

void add_cleanup_function(const std::string& name,
                          std::function<void()> cleanup_function) {
  get_cleanup_registry().emplace_back(name, cleanup_function);
}

void cleanup_all() {
  // it is not safe to call glog() any longer
  // deinitialize_onnxruntime_morphizen_ep might be called again.
  MY_LOG(1) << "cleanup_all() called";
  auto& cleanups = get_cleanup_registry();
  for (auto& cleanup : cleanups) {
    MY_LOG(1) << "cleanup function: " << cleanup.first;
    cleanup.second();
  }
  // g_at_exits.clear();
  cleanups.clear();
  MY_LOG(1) << "cleanup_all() done";
  // it is possible that the deinitialization is called multiple times, and
  // after glog is deconstructed it is not safe to call MY_LOG any longer
  ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE) = 0;
}
} // namespace morphizen
