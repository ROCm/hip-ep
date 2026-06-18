/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef INFERENCE_STATE_H
#define INFERENCE_STATE_H

#include "artifact_format.h" // ArtifactKind
#include "custom_op_mlir.hpp"
#include <memory>
#include <string>
#include <vector>

// Forward declare to avoid include order issues
namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir_compilation::customop {

class LoadedArtifact;

// Owns the loaded per-model artifact (via LoadedArtifact) that provides the
// `inference_*` entry points. The artifact is either an in-process ORC JIT
// (LLVM IR) or a morphizen::Plugin (native .dll/.so); both expose the
// identical 5-symbol C ABI, so the cached hot-path function pointers and
// Compute()/begin_compute()/cleanup paths are format-agnostic.
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
  // Create inference state from the per-model artifact stored in the
  // EPContext tar. `artifact_bytes` is the raw `.bc` (Bitcode) or native
  // `.dll`/`.so` (Native) blob; `kind` selects the loader. `fs` is the
  // morphizen FileSystem that resolves model constants and is forwarded
  // to `inference_init`. Logs FATAL and terminates on failure.
  static std::unique_ptr<InferenceState>
  create(const std::vector<uint8_t> &artifact_bytes, morphizen::FileSystem *fs,
         ArtifactKind kind = ArtifactKind::LLVM_IR);

  ~InferenceState();

  // Non-copyable, non-movable
  InferenceState(const InferenceState &) = delete;
  InferenceState &operator=(const InferenceState &) = delete;
  InferenceState(InferenceState &&) = delete;
  InferenceState &operator=(InferenceState &&) = delete;

  // Execute inference computation (classic 3-arg ABI).
  int compute(span_t *inputs, span_t *outputs) const;

  // Output-allocator mode: 2-arg inference_compute (state, inputs). Graph
  // outputs are allocated in-graph via the callback installed by
  // set_output_allocator(); there is no outputs span. The artifact exports
  // exactly one arity, fixed at compile time by use_output_allocator.
  int compute_with_output_allocator(span_t *inputs) const;

  // Install (allocator != nullptr) or clear (nullptr) the output allocator on
  // the loaded artifact's RuntimeState before compute_with_output_allocator().
  // Resolved once in the ctor. Fatal if called with a non-null allocator but
  // the artifact does not export the setter (a stale allocator-mode artifact
  // would otherwise crash with a null output buffer).
  void set_output_allocator(const output_allocator_t *allocator) const;

  // Invokes the optional `hipdnn_ep_runtime_begin_compute` hook to invalidate
  // per-forward-pass runtime caches (e.g. the GQA seqlens_k cache). create()
  // warns when the symbol is absent.
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
  // `artifact` owns the active backend (LLVM-IR JIT or native Plugin);
  // teardown (incl. any native temp file) is handled by LoadedArtifact.
  InferenceState(PrivateTag, void *state,
                 std::unique_ptr<LoadedArtifact> artifact);

private:
  // Resolve and cache the inference_* entry points from the loaded artifact.
  void resolveEntryPoints(const LoadedArtifact &artifact);

  // Opaque handle returned by inference_init()
  void *state_;
  // Must outlive `state_`: `inference_init` returned a pointer into memory
  // allocated by the loaded artifact.
  std::unique_ptr<LoadedArtifact> artifact_;

  // Entry points cached at construction so the per-Compute() (compute_fn_) and
  // teardown (cleanup_fn_) paths each pay one indirect call instead of an ORC
  // symbol lookup -- the lookup walks the JITDylib and every search generator
  // (tens of microseconds per token at 32-layer LLM decode). A missing symbol
  // leaves the pointer null; the error surfaces at first use (compute() /
  // ~InferenceState()) rather than at session creation.
  using ComputeFn = int (*)(void *, span_t *, span_t *);
  using ComputeAllocFn = int (*)(void *,
                                 span_t *); // 2-arg output-allocator ABI
  using CleanupFn = int (*)(void *);
  ComputeFn compute_fn_;
  ComputeAllocFn compute_alloc_fn_;
  CleanupFn cleanup_fn_;

  // Cached so begin_compute() is a single indirect call on the decode hot
  // path. Null when the module does not export the hook.
  using BeginComputeFn = void (*)(void *);
  BeginComputeFn begin_compute_fn_;

  // Cached hipdnn_ep_set_output_allocator (resolved once in the ctor, like
  // begin_compute_fn_). Null when the model.dll predates the export (classic
  // DLLs); only used in output-allocator mode.
  using SetOutputAllocatorFn = void (*)(void *, const output_allocator_t *);
  SetOutputAllocatorFn set_output_allocator_fn_;

  // Cached function pointer for hipdnn_ep_runtime_flush_op_profile. Same
  // contract as begin_compute_fn_: resolved once at session creation, null
  // when the symbol is absent (older DLLs), one cached indirect call when
  // present.
  using FlushOpProfileFn = void (*)(void *);
  FlushOpProfileFn flush_op_profile_fn_;

  // Optional debug CPU fallback: second arg is the C iface from
  // hipdnn_ep_runtime.h, kept as const void* here so this header does not
  // include that file (avoids typedef vs forward-declare clashes with MSVC).
  using SetCpuFallbackFn = void (*)(void *, const void *);
  SetCpuFallbackFn set_cpu_fallback_fn_{nullptr};
};

} // namespace mlir_compilation::customop

#endif
