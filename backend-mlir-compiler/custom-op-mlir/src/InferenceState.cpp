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
#include <utility>

#if defined(HIPDNN_EP_LINK_HIP_HOST)
#include <hip/hip_runtime.h>
#endif

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")
// S1: HIP-graph capture-safety probe. When >=1, compute() attempts a
// hipStreamBeginCapture/EndCapture around compute_fn_ once (per process) to
// learn whether the current decode is capture-clean, then either replays the
// captured graph (success) or runs eager (failure). Off (0) => no behavior
// change. Capture records but does NOT execute, so exactly one execution
// happens on either branch (no double state update).
DEF_ENV_PARAM(HIPDNN_EP_CAPTURE_PROBE, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation::customop {

namespace {
// Resolve inference_init from the loaded artifact, call it, and return the
// opaque state. FATALs (like the rest of create()) on a missing symbol or
// non-zero return.
void *runInit(const LoadedArtifact &artifact, morphizen::FileSystem *fs) {
  auto init_fn =
      artifact.get_method<int, void **, void *>(hipdnn::abi::kInferenceInit);
  if (!init_fn) {
    LOG(FATAL) << "inference_init not found in artifact.";
  }
  void *state = nullptr;
  int ret = init_fn(&state, static_cast<void *>(fs));
  if (ret != 0) {
    LOG(FATAL) << "inference_init() failed with code: " << ret;
  }
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
  set_decode_hint_fn_ =
      artifact.get_method<void, void *, int32_t, int32_t>(
          hipdnn::abi::kRuntimeSetDecodeHint);
  set_output_allocator_fn_ =
      artifact.get_method<void, void *, const output_allocator_t *>(
          hipdnn::abi::kSetOutputAllocator);
  flush_op_profile_fn_ =
      artifact.get_method<void, void *>(hipdnn::abi::kRuntimeFlushOpProfile);
}

InferenceState::InferenceState(PrivateTag, void *state,
                               std::unique_ptr<LoadedArtifact> artifact)
    : state_(state), artifact_(std::move(artifact)), compute_fn_(nullptr),
      cleanup_fn_(nullptr), begin_compute_fn_(nullptr),
      set_decode_hint_fn_(nullptr), set_output_allocator_fn_(nullptr),
      flush_op_profile_fn_(nullptr) {
  resolveEntryPoints(*artifact_);

  MY_LOG(2) << "begin_compute symbol "
            << (begin_compute_fn_ ? "resolved" : "not exported (no-op)");
  MY_LOG(2) << "set_output_allocator symbol "
            << (set_output_allocator_fn_ ? "resolved" : "not exported (no-op)");
  MY_LOG(2) << "flush_op_profile symbol "
            << (flush_op_profile_fn_ ? "resolved" : "not exported (no-op)");

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
    LOG(FATAL) << "InferenceState::create: failed to load " << kind_name
               << " artifact: " << load_err
               << " (per-model artifact malformed, or required external "
                  "symbols -- kernels / HIP runtime / CRT -- could not be "
                  "resolved in the EP DLL process).";
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
#if defined(HIPDNN_EP_LINK_HIP_HOST)
  // Scope the capture probe to decoder single-token steps. Attempting capture on
  // the vision/prefill sync path is fatal (S1); decode is the only candidate for
  // a capture-clean graph (esp. with HIPDNN_EP_DECODE_SKIP_SYNC removing the 3
  // readback syncs). Non-decode computes always run eager.
  if (ENV_PARAM(HIPDNN_EP_CAPTURE_PROBE) >= 1 && probe_is_decode_) {
    // Attempt capture only on a WARM decode step (after allocations/pool have
    // stabilized over the first few decode steps). Capturing on a cold step
    // hits hipMalloc (pool growth), which is illegal during capture. This
    // mirrors the warmup-before-capture flow of CUDA/ORT graph capture. Single
    // attempt to avoid repeated capture-invalidation poisoning the stream.
    static int decodeCount = 0;
    ++decodeCount;
    if (decodeCount == 4) {
      return computeWithCaptureProbe(inputs);
    }
  }
#endif
  return compute_fn_(state_, inputs);
}

#if defined(HIPDNN_EP_LINK_HIP_HOST)
// One-shot-per-process HIP-graph capture-safety probe (S1). Attempts to capture
// the compute_fn_ call into a graph. hipStreamBeginCapture RECORDS but does NOT
// execute kernels, so:
//   - success: instantiate + launch the graph once  -> exactly one execution
//   - failure: nothing executed during capture       -> run eager once
// Either branch executes compute exactly once (no double state update). The
// probe logs the verdict once, then stays out of the way (still runs each call,
// but capture-clean decode would be the S4 path; here it is diagnostic).
int InferenceState::computeWithCaptureProbe(span_t *inputs) const {
  static bool logged = false;
  static bool warned = false;
  if (!warned) {
    // Empirically (S1): if the decode still issues a mid-compute readback
    // (hipStreamSynchronize / D2H), that sync is FATAL under an active
    // hipStreamBeginCapture on HIP/Windows -- it crashes the process rather
    // than returning a soft error. So this probe is only safe once the decode
    // is capture-clean (readback_scalar==0, i.e. after S3). Enabling it before
    // then is expected to crash; that crash IS the "not capture-clean" signal.
    LOG(WARNING) << "[capture-probe] ENABLED: if the graph still has readbacks "
                    "this will CRASH (expected pre-S3 signal, not a bug)";
    warned = true;
  }
  void *raw = get_stream_raw();
  if (raw == nullptr) {
    if (!logged) {
      LOG(WARNING) << "[capture-probe] no stream on state; running eager";
      logged = true;
    }
    return compute_fn_(state_, inputs);
  }
  hipStream_t stream = reinterpret_cast<hipStream_t>(raw);
  hipGraph_t graph = nullptr;
  hipError_t berr =
      hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (berr != hipSuccess) {
    if (!logged) {
      LOG(WARNING) << "[capture-probe] hipStreamBeginCapture failed: "
                   << hipGetErrorString(berr) << "; running eager";
      logged = true;
    }
    return compute_fn_(state_, inputs);
  }
  int rc = compute_fn_(state_, inputs); // records only; does not execute
  hipError_t eerr = hipStreamEndCapture(stream, &graph);
  if (eerr == hipSuccess && graph != nullptr) {
    hipGraphExec_t exec = nullptr;
    hipError_t ierr = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (ierr == hipSuccess && exec != nullptr) {
      hipGraphLaunch(exec, stream);
      hipStreamSynchronize(stream);
      hipGraphExecDestroy(exec);
      hipGraphDestroy(graph);
      if (!logged) {
        LOG(INFO) << "[capture-probe] SUCCESS: decode IS capture-clean "
                     "(graph captured + replayed)";
        logged = true;
      }
      return rc;
    }
    hipGraphDestroy(graph);
    if (!logged) {
      LOG(WARNING) << "[capture-probe] EndCapture ok but Instantiate failed: "
                   << hipGetErrorString(ierr) << "; running eager";
      logged = true;
    }
    return compute_fn_(state_, inputs);
  }
  // Capture failed (e.g. a mid-compute readback/synchronize invalidated it).
  // Nothing executed during capture -> safe to run eager exactly once.
  if (graph != nullptr) {
    hipGraphDestroy(graph);
  }
  if (!logged) {
    LOG(INFO) << "[capture-probe] FAILED: decode is NOT capture-clean "
                 "(EndCapture err="
              << hipGetErrorString(eerr)
              << ") -- expected while readbacks remain; running eager";
    logged = true;
  }
  return compute_fn_(state_, inputs);
}
#endif

void InferenceState::set_output_allocator(
    const output_allocator_t *allocator) const {
  if (!set_output_allocator_fn_) {
    // Installing an allocator on a DLL that cannot accept it would leave
    // hip.alloc_output returning null -> guaranteed crash. Fail loudly with an
    // actionable message instead. Clearing (nullptr) on such a DLL is a no-op.
    if (allocator) {
      LOG(FATAL) << "Loaded model.dll does not export "
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

void InferenceState::set_decode_hint(bool is_decode, int32_t seqlens_k) const {
  if (set_decode_hint_fn_ && state_) {
    set_decode_hint_fn_(state_, is_decode ? 1 : 0, seqlens_k);
  }
}

void InferenceState::flush_op_profile() const {
  if (flush_op_profile_fn_ && state_) {
    flush_op_profile_fn_(state_);
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
