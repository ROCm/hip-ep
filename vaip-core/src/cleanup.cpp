/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./cleanup.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include <functional>
#include <glog/logging.h>
#include <utility>
#include <vector>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE) >= n)

namespace {
std::vector<std::pair<std::string, std::function<void()>>> g_at_exits;
} // namespace
namespace vaip_core {
void add_cleanup_function(const std::string& name,
                          std::function<void()> cleanup_function) {
  g_at_exits.emplace_back(name, cleanup_function);
}

void deinitialize_onnxruntime_vitisai_ep() {
  // it is not safe to call glog() any longer
  // deinitialize_onnxruntime_vitisai_ep might be called again.
  MY_LOG(1) << "deinitialize_onnxruntime_vitisai_ep() called";
  for (auto& cleanup : g_at_exits) {
    MY_LOG(1) << "cleanup function: " << cleanup.first;
    cleanup.second();
  }
  g_at_exits.clear();
  MY_LOG(1) << "deinitialize_onnxruntime_vitisai_ep() done";
  // it is possible that the deinitialization is called multiple times, and
  // after glog is deconstructed it is not safe to call MY_LOG any longer
  ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE) = 0;
}
} // namespace vaip_core
