/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "InferenceState.h"

#include "LoadedArtifact.h"
#include "hip/artifact_abi.h"
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <cstddef>
#include <cstdlib>
#include <glog/logging.h>
#include <stdexcept>
#include <utility>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation::customop {

namespace {
// Resolve inference_init from the ABI-validated artifact, call it, and return
// the opaque state. Throw on failure so CreateState can return an OrtStatus.
void *runInit(const LoadedArtifact &artifact, morphizen::FileSystem *fs) {
  auto init_fn =
      artifact.get_method<int, void **, void *>(hipdnn::abi::kInferenceInit);
  if (!init_fn)
    throw std::runtime_error("inference_init not found in artifact");
  void *state = nullptr;
  int ret = init_fn(&state, static_cast<void *>(fs));
  if (ret != 0)
    throw std::runtime_error("inference_init failed with code: " +
                             std::to_string(ret));
  return state;
}
} // namespace

// Resolve the hot-path entry points from the loaded artifact; LoadedArtifact
// exposes one get_method form over either backend.
void InferenceState::resolveEntryPoints(const LoadedArtifact &artifact) {
  // inference_compute has the 2-arg output-allocator ABI (state, inputs);
  // graph outputs are allocated in-graph via the output allocator callback.
  compute_fn_ = artifact.get_method<int, void *, span_t *>(
      hipdnn::abi::kInferenceCompute);
  cleanup_fn_ =
      artifact.get_method<int, void *>(hipdnn::abi::kInferenceCleanup);
  begin_compute_fn_ =
      artifact.get_method<void, void *>(hipdnn::abi::kRuntimeBeginCompute);
  set_output_allocator_fn_ =
      artifact.get_method<void, void *, const output_allocator_t *>(
          hipdnn::abi::kSetOutputAllocator);
  flush_op_profile_fn_ =
      artifact.get_method<void, void *>(hipdnn::abi::kRuntimeFlushOpProfile);
  // Perf-only hook; its name is a plain literal (not an artifact_abi.h
  // constant) since it is not part of the functional model ABI.
  add_cpu_profile_fn_ =
      artifact.get_method<void, void *, const char *, double, double>(
          "hipdnn_ep_runtime_add_cpu_profile");
}

InferenceState::InferenceState(PrivateTag, void *state,
                               std::unique_ptr<LoadedArtifact> artifact)
    : state_(state), artifact_(std::move(artifact)), compute_fn_(nullptr),
      cleanup_fn_(nullptr), begin_compute_fn_(nullptr),
      set_output_allocator_fn_(nullptr), flush_op_profile_fn_(nullptr),
      add_cpu_profile_fn_(nullptr) {
  resolveEntryPoints(*artifact_);

  MY_LOG(2) << "begin_compute symbol "
            << (begin_compute_fn_ ? "resolved" : "not exported (no-op)");
  MY_LOG(2) << "set_output_allocator symbol "
            << (set_output_allocator_fn_ ? "resolved" : "not exported (no-op)");
  MY_LOG(2) << "flush_op_profile symbol "
            << (flush_op_profile_fn_ ? "resolved" : "not exported (no-op)");
  MY_LOG(2) << "add_cpu_profile symbol "
            << (add_cpu_profile_fn_ ? "resolved" : "not exported (no-op)");

  // Without begin_compute, the GQA seqlens_k cache survives across
  // forward passes and decode returns token-1 values for tokens 2..N
  // (silently wrong logits). The cache is on by default, so warn unless
  // it was explicitly disabled.
  if (!begin_compute_fn_) {
    const char *env = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    const bool cache_enabled = !env || env[0] != '0';
    if (cache_enabled) {
      LOG(WARNING) << "GQA seqlens_k cache is enabled "
                   << "(HIPDNN_EP_GQA_CACHE_SEQLENS="
                   << (env ? env : "<unset, default 1>")
                   << "), but the loaded model artifact does not export "
                      "hipdnn_ep_runtime_begin_compute. Per-Compute() cache "
                      "invalidation will not happen and decode output will "
                      "be incorrect from token 2 onward. Either set "
                      "HIPDNN_EP_GQA_CACHE_SEQLENS=0 or rebuild the model "
                      "artifact with a runtime that exports the hook.";
    }
  }
}

std::unique_ptr<InferenceState>
InferenceState::create(const std::vector<uint8_t> &artifact_bytes,
                       morphizen::FileSystem *fs, ArtifactKind kind) {
  auto t0 = timing_now();
  auto t_prev = t0;

  const char *kind_name =
      (kind == ArtifactKind::LLVM_IR) ? "LLVM IR" : "native";
  MY_LOG(1) << "Loading inference artifact from EPContext (" << kind_name
            << ")...";

  // LoadedArtifact selects the backend (in-process ORC JIT for LLVM IR, or a
  // temp .dll/.so + morphizen::Plugin for native) and owns its teardown.
  std::string load_err;
  auto artifact = LoadedArtifact::createInMemory(artifact_bytes, kind,
                                                 "model_compiled", &load_err);
  if (!artifact) {
    throw std::runtime_error(
        std::string("InferenceState::create: failed to load ") + kind_name +
        " artifact: " + load_err +
        " (per-model artifact malformed, stale, or required external symbols "
        "-- kernels / HIP runtime / CRT -- could not be resolved in the EP "
        "DLL process)");
  }

  TIMING_LOG("[Session] LoadedArtifact::createInMemory (parse + JIT-init / "
             "Plugin load): %.3fs (%zu bytes)\n",
             record_elapsed(t_prev), artifact_bytes.size());

  // The inference_init lookup also triggers ORC's lazy codegen for the host
  // wrappers (LLVM IR), so this also covers first-symbol materialization.
  void *state = runInit(*artifact, fs);

  TIMING_LOG("[Session] inference_init (lookup + lazy codegen): %.3fs\n",
             record_elapsed(t_prev));
  TIMING_LOG("[Session] InferenceState::create total: %.3fs\n",
             elapsed_since(t0));

  MY_LOG(1) << "Inference state initialized (" << kind_name << ")";
  return std::make_unique<InferenceState>(PrivateTag{}, state,
                                          std::move(artifact));
}

InferenceState::~InferenceState() {
  MY_LOG(1) << "InferenceState destructor: cleaning up state";
  if (state_) {
    if (cleanup_fn_) {
      int ret = cleanup_fn_(state_);
      if (ret != 0) {
        LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
      }
    } else {
      LOG(WARNING)
          << "inference_cleanup symbol was not resolved at session "
             "creation; skipping teardown call (potential GPU resource "
             "leak in the loaded module).";
    }
    state_ = nullptr;
  }
  // artifact_ is destroyed next (LoadedArtifact tears down the ORC JIT or
  // unloads the Plugin, and removes any native temp file it created).
  MY_LOG(1) << "InferenceState destructor: loaded artifact will be destroyed "
               "next";
}

int InferenceState::compute(span_t *inputs) const {
  if (!compute_fn_) {
    LOG(ERROR) << "inference_compute symbol not resolved in loaded artifact";
    return -1;
  }
  return compute_fn_(state_, inputs);
}

void InferenceState::set_output_allocator(
    const output_allocator_t *allocator) const {
  if (!set_output_allocator_fn_) {
    // Installing an allocator on an artifact that cannot accept it would leave
    // hip.alloc_output returning null. Throw an actionable exception that the
    // Compute callback translates to OrtStatus.
    // Clearing (nullptr) on such an artifact is a no-op.
    if (allocator)
      throw std::runtime_error(
          "ABI-validated artifact does not export "
          "hipdnn_ep_set_output_allocator; regenerate the EPContext");
    return;
  }
  if (state_) {
    set_output_allocator_fn_(state_, allocator);
  }
}

void InferenceState::begin_compute() const {
  if (begin_compute_fn_ && state_) {
    begin_compute_fn_(state_);
  }
}

void InferenceState::flush_op_profile() const {
  if (flush_op_profile_fn_ && state_) {
    flush_op_profile_fn_(state_);
  }
}

void InferenceState::add_cpu_profile(const char *name, double cpu_start_us,
                                     double cpu_ms) const {
  if (add_cpu_profile_fn_ && state_) {
    add_cpu_profile_fn_(state_, name, cpu_start_us, cpu_ms);
  }
}

// First-field cast onto RuntimeState (see runtime_state_internal.h).
// EP DLL and runtime.bc come from the same source tree, so layout is
// consistent by construction; a layout drift would surface as absurd
// hipEventElapsedTime readings under HIPDNN_EP_PERF.
namespace {
struct RuntimeStateHead {
  void *stream;
};
static_assert(sizeof(void *) == 8, "expected 64-bit target");
static_assert(offsetof(RuntimeStateHead, stream) == 0,
              "RuntimeState layout invariant: stream must be first field");
} // namespace

void *InferenceState::get_stream_raw() const {
  return state_ ? reinterpret_cast<RuntimeStateHead *>(state_)->stream
                : nullptr;
}

} // namespace mlir_compilation::customop
