/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "InferenceState.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "../../common/temp_path.hpp"
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <utility>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace {
// Returns the platform-appropriate file extension for a compiled artifact.
// TODO: derive from artifact format stored in metadata when multiple formats
// are supported (e.g. ArtifactFormat::SharedLib → ".so" on Linux,
// ArtifactFormat::Portable → ".mlir")
std::string artifactExtension() {
#ifdef _WIN32
  return ".dll";
#else
  return ".so";
#endif
}

bool gpuRuntimeAllowed() {
  const char *value = std::getenv("HIPDNN_EP_ALLOW_GPU_RUNTIME");
  return value && std::strcmp(value, "1") == 0;
}
} // anonymous namespace

namespace mlir_compilation::customop {

InferenceState::InferenceState(PrivateTag, void *state,
                               std::unique_ptr<morphizen::Plugin> plugin,
                               const std::string &temp_dll_path)
    : state_(state), plugin_(std::move(plugin)), temp_dll_path_(temp_dll_path) {
}

std::unique_ptr<InferenceState>
InferenceState::create(const std::vector<uint8_t> &dll_bytes,
                       morphizen::FileSystem *fs) {
  MY_LOG(1) << "Loading inference plugin from memory...";

  if (!gpuRuntimeAllowed()) {
    LOG(FATAL) << "Refusing to load generated HIP EP runtime DLL without "
                  "HIPDNN_EP_ALLOW_GPU_RUNTIME=1. This path initializes "
                  "amdhip64/MIOpen/hipBLASLt and can trigger GPU watchdog "
                  "resets on display-attached Windows systems.";
  }

  auto t0 = timing_now();
  auto t_prev = t0;

  // Write DLL to temp file (morphizen::Plugin loads from file path)
  std::string dll_path =
      mlir_compiler_utils::generateTempPath(artifactExtension());
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

  TIMING_LOG("[Session] Write DLL to temp file: %.3fs (%zu bytes)\n",
             record_elapsed(t_prev), dll_bytes.size());

  // Load plugin using morphizen infrastructure (factory pattern)
  // Pass path without extension — Plugin::guess_name adds platform-correct
  // suffix
  std::string base_path =
      std::filesystem::path(dll_path).replace_extension("").string();
  auto plugin = morphizen::Plugin::create(base_path.c_str());

  // Check if plugin DLL loaded successfully
  if (!plugin) {
    LOG(FATAL)
        << "Failed to load DLL: " << dll_path
        << " - check that the file exists and all dependencies are available";
  }

  TIMING_LOG("[Session] Plugin::create (LoadLibrary): %.3fs\n",
             record_elapsed(t_prev));

  // inference_init(void** out_state, void* fs) — the DLL needs a FileSystem
  // to resolve and load model constants from the EPContext archive.
  auto init_fn = plugin->get_method<int, void **, void *>("inference_init");
  if (!init_fn) {
    LOG(FATAL) << "inference_init function not found in plugin: " << dll_path
               << " - DLL loaded successfully but symbol is missing";
  }

  TIMING_LOG("[Session] get_method (symbol lookup): %.3fs\n",
             record_elapsed(t_prev));

  void *state = nullptr;
  int ret = init_fn(&state, static_cast<void *>(fs));
  if (ret != 0) {
    LOG(FATAL) << "inference_init() failed with code: " << ret;
  }

  TIMING_LOG("[Session] inference_init: %.3fs\n", record_elapsed(t_prev));
  TIMING_LOG("[Session] InferenceState::create total: %.3fs\n",
             elapsed_since(t0));

  MY_LOG(1) << "Inference state initialized";

  return std::make_unique<InferenceState>(PrivateTag{}, state,
                                          std::move(plugin), dll_path);
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

} // namespace mlir_compilation::customop
