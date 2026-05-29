/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef INFERENCE_STATE_H
#define INFERENCE_STATE_H

#include "custom_op_mlir.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declare to avoid include order issues
namespace morphizen {
struct Plugin; // Must match definition in morphizen_plugin.hpp (struct, not
               // class)
class FileSystem;
} // namespace morphizen

namespace mlir_compilation::customop {

// Manages inference state and owns the plugin that provides inference
// functions. Uses morphizen::Plugin infrastructure for dynamic library loading.
class InferenceState {
  // Passkey tag for the public constructor below: external callers cannot
  // name this type (it is implicitly private under the `class` keyword), so
  // the constructor is effectively unreachable from outside the class even
  // though it is declared in the public section. Members and friends can
  // still construct one, which is what std::make_unique<InferenceState>(
  // PrivateTag{}, ...) needs inside create().
  //
  // Declaring the type here -- at the very top of the class body, before any
  // member function signature -- means name lookup for the constructor below
  // resolves to this inner type on both MSVC and GCC. Two earlier shapes
  // both broke one compiler:
  //   - Forward decl in `public:` + definition in `private:` is rejected by
  //     GCC with `error: ... PrivateTag redeclared with different access`.
  //   - Spelling the parameter as `struct PrivateTag` (an elaborated type
  //     specifier) makes MSVC introduce a brand-new outer-scope type, so
  //     the .cpp definition no longer matches the header signature and
  //     fails with C2511.
  // Defining it once, here, sidesteps both.
  struct PrivateTag {};

public:
  // Create inference state from DLL bytes.
  // fs: FileSystem for resolving model constants (passed to inference_init).
  // Logs FATAL and terminates on failure.
  static std::unique_ptr<InferenceState>
  create(const std::vector<uint8_t> &dll_bytes, morphizen::FileSystem *fs);

  ~InferenceState();

  // Non-copyable, non-movable
  InferenceState(const InferenceState &) = delete;
  InferenceState &operator=(const InferenceState &) = delete;
  InferenceState(InferenceState &&) = delete;
  InferenceState &operator=(InferenceState &&) = delete;

  // Execute inference computation
  int compute(span_t *inputs, span_t *outputs) const;

  // Mark the start of a new forward pass before inference_compute. If the
  // model.dll exports hipdnn_ep_runtime_begin_compute (resolved once in
  // create()) it is invoked to invalidate per-Compute() runtime caches
  // such as the GQA seqlens_k cache. On older model.dlls the symbol is
  // absent and this call is a no-op -- such DLLs must be paired with
  // HIPDNN_EP_GQA_CACHE_SEQLENS=0 (the cache is on by default; create()
  // logs a LOG(WARNING) when it detects the mismatch).
  void begin_compute() const;

  // Flush per-op profile (HIPDNN_EP_PERF). Called by the EP AFTER its
  // wall_ms timing window closes, so the resolve + std::map + fprintf cost
  // no longer pollutes Compute() latency. No-op when the model.dll predates
  // the export (per-op PERF block is silently skipped for such DLLs;
  // inference is unaffected). Symbol resolved once in create() so the call
  // is a single cached indirect dispatch.
  void flush_op_profile() const;

  // Diagnostic-only accessor: returns the hipStream_t used by
  // inference_compute, as a void*.  Relies on RuntimeState
  // (lib/Runtime/runtime_state_internal.h) keeping hipStream_t as its first
  // field; the cast is encapsulated in InferenceState.cpp with a static_assert
  // on pointer size.  Returning void* keeps hip headers out of this public
  // header.
  //
  // Intended for HIPDNN_EP_PERF instrumentation in MlirCustomOp::Compute().
  // Callers reinterpret_cast to hipStream_t (itself a void* on amdhip64).
  void *get_stream_raw() const;

  // Public constructor gated by PrivateTag (defined at the top of this
  // class). Use the create() factory instead -- external callers cannot
  // construct a PrivateTag and therefore cannot call this constructor.
  InferenceState(PrivateTag, void *state,
                 std::unique_ptr<morphizen::Plugin> plugin,
                 const std::string &temp_dll_path);

private:
  // Opaque handle returned by inference_init()
  void *state_;

  // Owned plugin - must outlive state_
  std::unique_ptr<morphizen::Plugin> plugin_;

  // Temporary DLL file path (deleted in destructor)
  std::string temp_dll_path_;

  // Cached function pointer for hipdnn_ep_runtime_begin_compute. Resolved
  // once in create() so begin_compute() avoids a per-call GetProcAddress /
  // dlsym round-trip on the decode hot path. Null when the model.dll
  // predates the export.
  using BeginComputeFn = void (*)(void *);
  BeginComputeFn begin_compute_fn_;

  // Cached function pointer for hipdnn_ep_runtime_flush_op_profile. Same
  // contract as begin_compute_fn_: resolved once at session creation, null
  // when the symbol is absent (older DLLs), one cached indirect call when
  // present.
  using FlushOpProfileFn = void (*)(void *);
  FlushOpProfileFn flush_op_profile_fn_;
};

} // namespace mlir_compilation::customop

#endif
