/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "InferenceState.h"

#include "BitcodeJIT.h"
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <cstddef>
#include <cstdlib>
#include <glog/logging.h>
#include <utility>

// Environment parameters (global scope, before namespace).
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation::customop {

InferenceState::InferenceState(PrivateTag, void *state,
                               std::unique_ptr<BitcodeJIT> jit)
    : state_(state), jit_(std::move(jit)), begin_compute_fn_(nullptr) {
  // Resolve the optional per-Compute() cache-invalidation hook once at
  // session creation so begin_compute() is a single indirect call. If
  // the JIT module omits the symbol (older runtime.bc), we leave the
  // pointer null and begin_compute() becomes a no-op.
  if (jit_) {
    begin_compute_fn_ =
        jit_->get_method<void, void *>("hipdnn_ep_runtime_begin_compute");
    MY_LOG(2) << "begin_compute symbol "
              << (begin_compute_fn_ ? "resolved" : "not exported (no-op)");
  }
  // Safety net: warn loudly when the seqlens_k cache is effectively on
  // but the model bitcode predates the begin_compute export. Without
  // the invalidation hook the cache would survive across forward passes
  // and the gqa.cpp readback would return token-1 values for tokens
  // 2..N, producing silently wrong logits. The cache defaults to on, so
  // this fires unless the user has explicitly set
  // HIPDNN_EP_GQA_CACHE_SEQLENS=0. Detected once at session creation so
  // the user gets an actionable signal before observing decode output
  // corruption.
  if (!begin_compute_fn_) {
    const char *env = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    const bool cache_enabled = !env || env[0] != '0';
    if (cache_enabled) {
      LOG(WARNING) << "GQA seqlens_k cache is enabled "
                   << "(HIPDNN_EP_GQA_CACHE_SEQLENS="
                   << (env ? env : "<unset, default 1>")
                   << "), but the loaded model bitcode does not export "
                      "hipdnn_ep_runtime_begin_compute. Per-Compute() cache "
                      "invalidation will not happen and decode output will "
                      "be incorrect from token 2 onward. Either set "
                      "HIPDNN_EP_GQA_CACHE_SEQLENS=0 or rebuild the model "
                      "bitcode with a runtime.bc that exports the hook.";
    }
  }
}

std::unique_ptr<InferenceState>
InferenceState::create(const std::vector<uint8_t> &bitcode_bytes,
                       morphizen::FileSystem *fs) {
  MY_LOG(1) << "JIT-loading inference bitcode from EPContext...";

  auto t0 = timing_now();
  auto t_prev = t0;

  // Hand the per-model bitcode to ORC LLJIT. BitcodeJIT installs a
  // process-wide symbol search generator so external references in the
  // bitcode (hip_*_kernel_launch_*, libamdhip64, MIOpen, hipblaslt, CRT)
  // are resolved against modules already loaded by ORT into the host
  // process. See BitcodeJIT.h for the full resolution model.
  auto jit = BitcodeJIT::create(bitcode_bytes, "model_compiled");
  if (!jit) {
    LOG(FATAL) << "BitcodeJIT::create failed -- per-model bitcode is "
                  "malformed or required external symbols (kernels / HIP "
                  "runtime / CRT) could not be resolved in the EP DLL "
                  "process.";
  }

  TIMING_LOG("[Session] BitcodeJIT::create (parse + JIT-init): %.3fs (%zu "
             "bytes)\n",
             record_elapsed(t_prev), bitcode_bytes.size());

  // Resolve the model entry points. `inference_init` does the heavy
  // lifting (allocates RuntimeState, opens MIOpen/hipBLASLt handles,
  // streams constants from `fs` into GPU memory). Looking the symbol up
  // here also triggers ORC's lazy codegen for the host-side host
  // wrappers, so the first GetProcAddress-equivalent path is paid once
  // per session.
  auto init_fn = jit->get_method<int, void **, void *>("inference_init");
  if (!init_fn) {
    LOG(FATAL) << "inference_init function not found in JIT module -- "
                  "bitcode was loaded but the expected entry-point symbol "
                  "is missing.";
  }

  TIMING_LOG("[Session] get_method (symbol lookup + lazy codegen): %.3fs\n",
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

  return std::make_unique<InferenceState>(PrivateTag{}, state, std::move(jit));
}

InferenceState::~InferenceState() {
  MY_LOG(1) << "InferenceState destructor: cleaning up state";
  if (state_ && jit_) {
    auto cleanup_fn = jit_->get_method<int, void *>("inference_cleanup");
    if (cleanup_fn) {
      int ret = cleanup_fn(state_);
      if (ret != 0) {
        LOG(WARNING) << "inference_cleanup() failed with code: " << ret;
      }
    }
    state_ = nullptr;
  }
  MY_LOG(1) << "InferenceState destructor: JIT will be destroyed next";
  // jit_ destructor unmaps the JIT'd code pages; safe to do after
  // inference_cleanup because all GPU resources held inside the JIT'd
  // RuntimeState have been released above.
}

int InferenceState::compute(span_t *inputs, span_t *outputs) const {
  auto compute_fn =
      jit_->get_method<int, void *, span_t *, span_t *>("inference_compute");
  if (!compute_fn) {
    LOG(ERROR) << "inference_compute function not found in JIT module";
    return -1;
  }
  return compute_fn(state_, inputs, outputs);
}

void InferenceState::begin_compute() const {
  if (begin_compute_fn_ && state_) {
    begin_compute_fn_(state_);
  }
}

// ----------------------------------------------------------------------------
// Diagnostic-only stream accessor.
//
// state_ is the opaque handle returned by inference_init() in the JIT
// module. That code (compiled from MLIR + runtime.bc) allocates a
// RuntimeState (see lib/Runtime/runtime_state_internal.h) and returns it
// as void*. RuntimeState's first field is `hipStream_t stream`, so a
// first-field cast gives us the stream without a codegen change or a
// new exported symbol.
//
// The EP DLL and the JIT'd model's linked-in runtime.bc are built from
// the same commit of the same tree, so layout is consistent by
// construction. If RuntimeState ever gains a field before `stream`, this
// returns garbage silently; we'd notice via absurd hipEventElapsedTime
// readings. Acceptable for a diagnostic (HIPDNN_EP_PERF) hook.
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
