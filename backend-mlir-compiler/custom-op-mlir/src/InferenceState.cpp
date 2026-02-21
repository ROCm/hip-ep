/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "InferenceState.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <utility>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

InferenceState::InferenceState(void *state,
                               std::unique_ptr<morphizen::Plugin> plugin,
                               const std::string &temp_dll_path)
    : state_(state), plugin_(std::move(plugin)), temp_dll_path_(temp_dll_path) {
}

std::unique_ptr<InferenceState>
InferenceState::create(const std::vector<uint8_t> &dll_bytes) {
  MY_LOG(1) << "Loading inference plugin from memory...";

  // Write DLL to temp file (morphizen::Plugin loads from file path)
  char temp_path[L_tmpnam];
  if (!std::tmpnam(temp_path)) {
    LOG(FATAL) << "Failed to generate temporary DLL path";
  }

  std::string dll_path = std::string(temp_path) + ".dll";
  MY_LOG(2) << "Temporary DLL path: " << dll_path;

  // Write DLL to temp file
  {
    std::ofstream dll_out(dll_path, std::ios::binary);
    if (!dll_out) {
      LOG(FATAL) << "Failed to create temporary DLL file: " << dll_path;
    }
    dll_out.write(reinterpret_cast<const char *>(dll_bytes.data()),
                  dll_bytes.size());
    dll_out.close();
  }

  // Load plugin using morphizen infrastructure (factory pattern)
  auto plugin = morphizen::Plugin::create(dll_path.c_str());

  // Check if plugin DLL loaded successfully
  if (!plugin) {
    LOG(FATAL)
        << "Failed to load DLL: " << dll_path
        << " - check that the file exists and all dependencies are available";
  }

  // Get init function and call it
  // NOTE: Keep temp DLL file until plugin is destroyed
  auto init_fn = plugin->get_method<int, void **>("inference_init");
  if (!init_fn) {
    LOG(FATAL) << "inference_init function not found in plugin: " << dll_path
               << " - DLL loaded successfully but symbol is missing";
  }

  void *state = nullptr;
  int ret = init_fn(&state);
  if (ret != 0) {
    LOG(FATAL) << "inference_init() failed with code: " << ret;
  }

  MY_LOG(1) << "Inference state initialized";

  return std::unique_ptr<InferenceState>(
      new InferenceState(state, std::move(plugin), dll_path));
}

InferenceState::~InferenceState() {
  MY_LOG(1) << "InferenceState destructor: cleaning up state";
  if (state_ && plugin_) {
    auto cleanup_fn = plugin_->get_method<int, void *>("inference_cleanup");
    if (cleanup_fn) {
      int ret = cleanup_fn(state_);
      if (ret != 0) {
        LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
      }
    }
    state_ = nullptr;
  }
  MY_LOG(1) << "InferenceState destructor: plugin will be destroyed next";
  // plugin_ destructor runs automatically after this

  // Delete temporary DLL file after plugin is destroyed
  if (!temp_dll_path_.empty()) {
    std::remove(temp_dll_path_.c_str());
    MY_LOG(2) << "Deleted temporary DLL: " << temp_dll_path_;
  }
}

int InferenceState::compute(span_t *inputs, span_t *outputs) const {
  auto compute_fn =
      plugin_->get_method<int, void *, span_t *, span_t *>("inference_compute");
  if (!compute_fn) {
    LOG(ERROR) << "inference_compute function not found in plugin";
    return -1;
  }
  return compute_fn(state_, inputs, outputs);
}

} // namespace customop
} // namespace mlir_compilation
