/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipInferenceState.h"

#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <utility>

DEF_ENV_PARAM(HIP_COMPILER_DEBUG, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(HIP_COMPILER_DEBUG) >= n)

namespace {
std::string artifactExtension() {
#ifdef _WIN32
  return ".dll";
#else
  return ".so";
#endif
}

std::string generateTempPath(const std::string &suffix) {
  auto tmp = std::filesystem::temp_directory_path();
  std::string name =
      "hip_model_" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
      suffix;
  return (tmp / name).string();
}
} // namespace

namespace hip_compilation::customop {

HipInferenceState::HipInferenceState(PrivateTag, void *state,
                                     std::unique_ptr<morphizen::Plugin> plugin,
                                     const std::string &temp_dll_path)
    : state_(state), plugin_(std::move(plugin)), temp_dll_path_(temp_dll_path) {
}

std::unique_ptr<HipInferenceState>
HipInferenceState::create(const std::vector<uint8_t> &dll_bytes) {
  MY_LOG(1) << "Loading HIP inference plugin from memory...";

  std::string dll_path = generateTempPath(artifactExtension());
  MY_LOG(2) << "Temporary DLL path: " << dll_path;

  {
    std::ofstream out(dll_path, std::ios::binary);
    if (!out) {
      LOG(FATAL) << "Failed to create temporary DLL file: " << dll_path;
    }
    out.write(reinterpret_cast<const char *>(dll_bytes.data()),
              dll_bytes.size());
  }

  std::string base_path =
      std::filesystem::path(dll_path).replace_extension("").string();
  auto plugin = morphizen::Plugin::create(base_path.c_str());
  if (!plugin) {
    LOG(FATAL) << "Failed to load DLL: " << dll_path;
  }

  auto init_fn = plugin->get_method<int, void **>("inference_init");
  if (!init_fn) {
    LOG(FATAL) << "inference_init not found in plugin: " << dll_path;
  }

  void *state = nullptr;
  int ret = init_fn(&state);
  if (ret != 0) {
    LOG(FATAL) << "inference_init() failed with code: " << ret;
  }

  MY_LOG(1) << "HIP inference state initialized";
  return std::make_unique<HipInferenceState>(PrivateTag{}, state,
                                             std::move(plugin), dll_path);
}

HipInferenceState::~HipInferenceState() {
  MY_LOG(1) << "HipInferenceState destructor";
  if (state_ && plugin_) {
    auto cleanup_fn = plugin_->get_method<int, void *>("inference_cleanup");
    if (cleanup_fn) {
      int ret = cleanup_fn(state_);
      if (ret != 0)
        LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
    }
    state_ = nullptr;
  }

  plugin_.reset();

  if (!temp_dll_path_.empty()) {
    std::remove(temp_dll_path_.c_str());
    MY_LOG(2) << "Deleted temporary DLL: " << temp_dll_path_;
  }
}

int HipInferenceState::compute(span_t *inputs, span_t *outputs) const {
  auto compute_fn =
      plugin_->get_method<int, void *, span_t *, span_t *>("inference_compute");
  if (!compute_fn) {
    LOG(ERROR) << "inference_compute not found in plugin";
    return -1;
  }
  return compute_fn(state_, inputs, outputs);
}

} // namespace hip_compilation::customop
