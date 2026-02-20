/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef INFERENCE_STATE_H
#define INFERENCE_STATE_H

#include "LlvmJitLoader.h"
#include "NativeDllLoader.h"
#include <optional>
#include <variant>

namespace mlir_compilation {
namespace customop {

// Manages inference state and owns the artifact (DLL or JIT) that provides
// the inference functions. This ensures the artifact remains valid for the
// lifetime of the inference state.
//
// OWNERSHIP: InferenceState owns the artifact and guarantees correct
// destruction order: inference state cleanup runs first, then artifact
// unload. This prevents use-after-free when calling cleanup_fn.
class InferenceState {
public:
  // Create inference state with artifact ownership.
  // Takes artifact by value to transfer ownership from caller.
  static std::optional<InferenceState>
  create(std::variant<DllHandle, JitHandle> artifact);

  ~InferenceState();

  // Non-copyable: Cannot duplicate owned artifact or inference state pointer.
  // The artifact (DLL/JIT) and state handle are unique resources.
  InferenceState(const InferenceState &) = delete;
  InferenceState &operator=(const InferenceState &) = delete;

  // Movable: Required for std::optional<InferenceState> and factory pattern.
  // Move semantics allow transferring ownership from create() to caller.
  // CRITICAL: These are needed because:
  //   1. std::variant<DllHandle, JitHandle> requires movable types
  //   (DllHandle/JitHandle are move-only)
  //   2. std::optional<InferenceState> requires movable types
  //   3. Factory pattern (create() returning optional) requires move semantics
  InferenceState(InferenceState &&other) noexcept;
  InferenceState &operator=(InferenceState &&other) noexcept;

  // Execute inference computation.
  // ENCAPSULATION: Internally manages state handle and artifact function calls.
  // Returns 0 on success, non-zero error code on failure.
  int compute(span_t *inputs, span_t *outputs) const;

private:
  // Private constructor: Use create() factory method instead.
  // Takes ownership of artifact via move.
  InferenceState(void *state, std::variant<DllHandle, JitHandle> artifact);

  // Opaque handle returned by inference_init()
  void *state_;

  // Owned artifact (DLL or JIT compiled code).
  // CRITICAL: This must outlive state_ because cleanup_fn is obtained from
  // artifact_.functions(). Destruction order is guaranteed: state_ cleanup
  // happens in destructor body before artifact_ is destroyed.
  std::variant<DllHandle, JitHandle> artifact_;
};

} // namespace customop
} // namespace mlir_compilation

#endif
