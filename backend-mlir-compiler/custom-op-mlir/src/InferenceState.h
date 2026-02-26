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
}

namespace mlir_compilation::customop {

// Manages inference state and owns the plugin that provides inference
// functions. Uses morphizen::Plugin infrastructure for dynamic library loading.
class InferenceState {
public:
  // Create inference state from DLL bytes
  // Logs FATAL and terminates on failure
  static std::unique_ptr<InferenceState>
  create(const std::vector<uint8_t> &dll_bytes);

  ~InferenceState();

  // Non-copyable, non-movable
  InferenceState(const InferenceState &) = delete;
  InferenceState &operator=(const InferenceState &) = delete;
  InferenceState(InferenceState &&) = delete;
  InferenceState &operator=(InferenceState &&) = delete;

  // Execute inference computation
  int compute(span_t *inputs, span_t *outputs) const;

  // Passkey tag — only create() can construct one; enables std::make_unique
  // with an otherwise-private constructor.
  struct PrivateTag {};

  // Public constructor gated by PrivateTag: use create() factory instead.
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
};

} // namespace mlir_compilation::customop

#endif
