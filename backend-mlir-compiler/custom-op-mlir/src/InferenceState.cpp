/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "InferenceState.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>
#include <utility>

// Component headers
#include "LlvmJitLoader.h"
#include "NativeDllLoader.h"

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

InferenceState::InferenceState(void *state,
                               std::variant<DllHandle, JitHandle> artifact)
    : state_(state), artifact_(std::move(artifact)) {}

std::optional<InferenceState>
InferenceState::create(std::variant<DllHandle, JitHandle> artifact) {

  // Extract functions from artifact before calling init
  const DllFunctions *functions_ptr = std::visit(
      [](auto &handle) -> const DllFunctions * { return &handle.functions(); },
      artifact);

  if (!functions_ptr || !functions_ptr->init) {
    LOG(WARNING) << "inference_init function is null";
    return std::nullopt;
  }

  void *state = nullptr;
  int ret = functions_ptr->init(&state);
  if (ret != 0) {
    LOG(WARNING) << "inference_init() failed with code: " << ret;
    return std::nullopt;
  }

  MY_LOG(1) << "Inference state initialized";

  return InferenceState(state, std::move(artifact));
}

InferenceState::~InferenceState() {
  MY_LOG(1) << "InferenceState destructor: cleaning up state";
  if (state_) {
    // Extract cleanup function before calling it
    const DllFunctions *functions_ptr = std::visit(
        [](auto &handle) -> const DllFunctions * {
          return &handle.functions();
        },
        artifact_);

    if (functions_ptr && functions_ptr->cleanup) {
      int ret = functions_ptr->cleanup(state_);
      if (ret != 0) {
        LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
      }
    }
    state_ = nullptr;
  }
  MY_LOG(1) << "InferenceState destructor: artifact will be destroyed next";
  // artifact_ destructor runs automatically after this
}

InferenceState::InferenceState(InferenceState &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      artifact_(std::move(other.artifact_)) {}

InferenceState &InferenceState::operator=(InferenceState &&other) noexcept {
  if (this != &other) {
    if (state_) {
      const DllFunctions *functions_ptr = std::visit(
          [](auto &handle) -> const DllFunctions * {
            return &handle.functions();
          },
          artifact_);

      if (functions_ptr && functions_ptr->cleanup) {
        int ret = functions_ptr->cleanup(state_);
        if (ret != 0) {
          LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
        }
      }
    }
    state_ = other.state_;
    artifact_ = std::move(other.artifact_);
    other.state_ = nullptr;
  }
  return *this;
}

int InferenceState::compute(span_t *inputs, span_t *outputs) const {
  // Get compute function from owned artifact
  const DllFunctions &functions = std::visit(
      [](const auto &handle) -> const DllFunctions & {
        return handle.functions();
      },
      artifact_);

  // Call inference_compute with our state handle
  return functions.compute(state_, inputs, outputs);
}

} // namespace customop
} // namespace mlir_compilation
