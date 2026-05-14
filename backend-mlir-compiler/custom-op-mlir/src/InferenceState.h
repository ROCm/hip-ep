/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef INFERENCE_STATE_H
#define INFERENCE_STATE_H

#include "custom_op_mlir.hpp"
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir_compilation::customop {

class BitcodeJIT;

// Manages inference state and owns the JIT-compiled bitcode module that
// provides the per-model `inference_*` entry points.
//
// The compiled artifact carried inside an EPContext tar (`model_compiled`
// entry) is LLVM bitcode -- the host-side runtime is statically merged in
// at compile time via `LLVMBackend::linkRuntimeModule`, GPU device code
// lives inside the signed EP DLL as fatbin sections from
// `hip_custom_kernels`. The artifact is therefore data, not code, and is
// JIT-compiled in-process via `BitcodeJIT` (ORC `LLJIT`). The previous
// "drop unsigned PE to TEMP, LoadLibrary, delete" path is gone, removing
// the WDAC / EDR reflective-DLL hazard.
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
  // Create inference state from the per-model bitcode payload stored in
  // the EPContext tar. `bitcode_bytes` is the raw `.bc` blob. `fs` is the
  // morphizen FileSystem that resolves model constants and is forwarded
  // to `inference_init`. Logs FATAL and terminates on failure.
  static std::unique_ptr<InferenceState>
  create(const std::vector<uint8_t> &bitcode_bytes, morphizen::FileSystem *fs);

  ~InferenceState();

  // Non-copyable, non-movable (the JIT owns generated machine code whose
  // page mappings cannot be moved without invalidating function pointers).
  InferenceState(const InferenceState &) = delete;
  InferenceState &operator=(const InferenceState &) = delete;
  InferenceState(InferenceState &&) = delete;
  InferenceState &operator=(InferenceState &&) = delete;

  // Execute inference computation.
  int compute(span_t *inputs, span_t *outputs) const;

  // Mark the start of a new forward pass before inference_compute. When
  // the JIT module exports `hipdnn_ep_runtime_begin_compute` (resolved
  // once in create()) the symbol is invoked to invalidate per-Compute()
  // runtime caches such as the GQA seqlens_k cache. The hook is required
  // for the seqlens_k cache to be correct; create() logs a LOG(WARNING)
  // when the symbol is absent.
  void begin_compute() const;

  // Diagnostic-only accessor: returns the hipStream_t used by
  // inference_compute, as a void*. Relies on RuntimeState
  // (lib/Runtime/runtime_state_internal.h) keeping hipStream_t as its
  // first field; the cast is encapsulated in InferenceState.cpp with a
  // static_assert on pointer size. Returning void* keeps hip headers out
  // of this public header.
  //
  // Intended for HIPDNN_EP_PERF instrumentation in MlirCustomOp::Compute().
  // Callers reinterpret_cast to hipStream_t (itself a void* on amdhip64).
  void *get_stream_raw() const;

  // Public constructor gated by PrivateTag (defined at the top of this
  // class). Use the create() factory instead -- external callers cannot
  // construct a PrivateTag and therefore cannot call this constructor.
  InferenceState(PrivateTag, void *state, std::unique_ptr<BitcodeJIT> jit);

private:
  // Opaque handle returned by inference_init().
  void *state_;

  // Owns the JIT'd machine code that provides inference_init / compute /
  // cleanup / begin_compute. Must outlive `state_` because the pointer
  // returned by inference_init references memory allocated by JIT'd code.
  std::unique_ptr<BitcodeJIT> jit_;

  // Cached function pointer for hipdnn_ep_runtime_begin_compute. Resolved
  // once in create() so begin_compute() avoids a per-call symbol-table
  // round-trip on the decode hot path. Null when the module does not
  // export the hook.
  using BeginComputeFn = void (*)(void *);
  BeginComputeFn begin_compute_fn_;
};

} // namespace mlir_compilation::customop

#endif
