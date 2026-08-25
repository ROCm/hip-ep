/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Minimal bf16 <-> float helpers shared by alloc_probe and npu_probe, so both
// programs write/verify the exact same "recognizable pattern" without pulling
// in hybrid-llm's ryzenai::onnx_utils bfloat16_to_float_single /
// float_to_bfloat16 (common.cpp), which drag in the rest of that tree.

#include <cstdint>
#include <cstring>

namespace npu_spike {

inline uint16_t FloatToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Truncate, not round-to-nearest -- adequate for a spike that only checks
  // "did the NPU/GPU touch this buffer", not bf16 rounding correctness.
  return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16ToFloat(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace npu_spike
