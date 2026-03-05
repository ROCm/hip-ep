/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_INFERENCE_STATE_H
#define HIP_INFERENCE_STATE_H

#include "custom_op_hip.hpp"
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
struct Plugin;
}

namespace hip_compilation::customop {

/// Manages the lifecycle of a compiled HIP model DLL.
/// Owns the loaded plugin and the opaque inference state handle.
class HipInferenceState {
public:
  static std::unique_ptr<HipInferenceState>
  create(const std::vector<uint8_t> &dll_bytes);

  ~HipInferenceState();

  HipInferenceState(const HipInferenceState &) = delete;
  HipInferenceState &operator=(const HipInferenceState &) = delete;
  HipInferenceState(HipInferenceState &&) = delete;
  HipInferenceState &operator=(HipInferenceState &&) = delete;

  int compute(span_t *inputs, span_t *outputs) const;

  struct PrivateTag;
  HipInferenceState(PrivateTag, void *state,
                    std::unique_ptr<morphizen::Plugin> plugin,
                    const std::string &temp_dll_path);

private:
  struct PrivateTag {};
  void *state_;
  std::unique_ptr<morphizen::Plugin> plugin_;
  std::string temp_dll_path_;
};

} // namespace hip_compilation::customop

#endif
