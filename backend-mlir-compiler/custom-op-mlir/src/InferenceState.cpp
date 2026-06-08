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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
} // anonymous namespace

namespace mlir_compilation::customop {

InferenceState::InferenceState(PrivateTag, void *state,
                               std::unique_ptr<morphizen::Plugin> plugin,
                               const std::string &temp_dll_path)
    : state_(state), plugin_(std::move(plugin)), temp_dll_path_(temp_dll_path),
      begin_compute_fn_(nullptr), set_output_allocator_fn_(nullptr) {
  // Cache the begin_compute symbol so the per-Compute() invocation is a
  // single indirect call. Older model.dlls do not export this symbol; in
  // that case we leave begin_compute_fn_ null and begin_compute() becomes
  // a no-op.
  if (plugin_) {
    begin_compute_fn_ =
        plugin_->get_method<void, void *>("hipdnn_ep_runtime_begin_compute");
    MY_LOG(2) << "begin_compute symbol "
              << (begin_compute_fn_ ? "resolved" : "not exported (no-op)");
    // Output-allocator setter: optional, like begin_compute. Only invoked in
    // allocator mode; classic DLLs leave this null.
    set_output_allocator_fn_ =
        plugin_->get_method<void, void *, const output_allocator_t *>(
            "hipdnn_ep_set_output_allocator");
    MY_LOG(2) << "set_output_allocator symbol "
              << (set_output_allocator_fn_ ? "resolved"
                                           : "not exported (no-op)");
  }
  // Safety net: warn loudly when the seqlens_k cache is effectively on
  // but the model.dll predates the begin_compute export. Without the
  // invalidation hook the cache would survive across forward passes and
  // the gqa.cpp readback would return token-1 values for tokens 2..N,
  // producing silently wrong logits. The cache defaults to on, so this
  // fires unless the user has explicitly set HIPDNN_EP_GQA_CACHE_SEQLENS=0.
  // Detected once at session creation so the user gets an actionable
  // signal before observing decode output corruption.
  if (!begin_compute_fn_) {
    const char *env = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    const bool cache_enabled = !env || env[0] != '0';
    if (cache_enabled) {
      LOG(WARNING) << "GQA seqlens_k cache is enabled "
                   << "(HIPDNN_EP_GQA_CACHE_SEQLENS="
                   << (env ? env : "<unset, default 1>")
                   << "), but the loaded model.dll does not export "
                      "hipdnn_ep_runtime_begin_compute. Per-Compute() cache "
                      "invalidation will not happen and decode output will be "
                      "incorrect from token 2 onward. Either set "
                      "HIPDNN_EP_GQA_CACHE_SEQLENS=0 or rebuild the model.dll "
                      "with a runtime that exports the hook.";
    }
  }
}

std::unique_ptr<InferenceState>
InferenceState::create(const std::vector<uint8_t> &dll_bytes,
                       morphizen::FileSystem *fs) {
  MY_LOG(1) << "Loading inference plugin from memory...";

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

int InferenceState::compute_allocator(span_t *inputs) const {
  // Same symbol as compute(), resolved with the 2-arg ABI. The DLL exports
  // exactly one arity (fixed at compile time by use_output_allocator), so this
  // is only ever called against an allocator-mode DLL.
  auto compute_fn =
      plugin_->get_method<int, void *, span_t *>("inference_compute");
  if (!compute_fn) {
    LOG(ERROR) << "inference_compute (2-arg allocator ABI) not found in plugin";
    return -1;
  }
  return compute_fn(state_, inputs);
}

void InferenceState::set_output_allocator(
    const output_allocator_t *allocator) const {
  if (!set_output_allocator_fn_) {
    // Installing an allocator on a DLL that cannot accept it would leave
    // hip.alloc_output returning null -> guaranteed crash. Fail loudly with an
    // actionable message instead. Clearing (nullptr) on such a DLL is a no-op.
    if (allocator) {
      LOG(FATAL) << "Model was compiled in output-allocator mode but the "
                    "loaded model.dll does not export "
                    "hipdnn_ep_set_output_allocator. The DLL is stale; delete "
                    "%TEMP%/morphizen_mlir_* (or regenerate the EPContext) and "
                    "rebuild so the linked runtime exports the setter.";
    }
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

// ----------------------------------------------------------------------------
// Diagnostic-only stream accessor.
//
// state_ is the opaque handle returned by inference_init() in the JIT-compiled
// model.dll.  That DLL's implementation of inference_init allocates a
// RuntimeState (see lib/Runtime/runtime_state_internal.h) and returns it as
// void*.  RuntimeState's first field is `hipStream_t stream`, so a first-field
// cast gives us the stream without a codegen change or a DLL export.
//
// The EP DLL and the JIT model.dll's linked-in runtime bitcode are built from
// the same commit of the same tree, so layout is consistent by construction.
// If RuntimeState ever gains a field before `stream`, this returns garbage
// silently; we'd notice via absurd hipEventElapsedTime readings.  Acceptable
// for a diagnostic (HIPDNN_EP_PERF) instrumentation.
// ----------------------------------------------------------------------------
namespace {
struct RuntimeStateHead {
  void *stream; // mirrors RuntimeState::stream (hipStream_t is pointer-sized)
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
