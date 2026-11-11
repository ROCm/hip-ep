/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstdint>
#include <string_view>

namespace hipdnn_ep_test {
namespace timing {

struct DeviceProfile {
  const char* name;
  uint32_t batch_call_cap;
  double screen_target_ms = 100.0;
  double screen_soft_limit_ms = 200.0;
};

// A normal profile always exists for a new GPU, runtime, entry or cache mode.
// Profiles change the safe event-batch size, never guess a device's latency.
// Actual total iterations still adapt to this device's measured GPU duration.
inline DeviceProfile deviceProfile(std::string_view architecture, int runtime,
                                   std::string_view entry, bool warm,
                                   std::string_view shape = {}) {
  const DeviceProfile normal{"normal", 64};
  if (entry != "matmul_nbits_fp16_wu4" || warm) return normal;
  if (shape != "M1_N2048_K2048_gs128" &&
      shape != "M128_N2048_K2048_gs128") return normal;
  // MatMulNBits UT qualification on 2026-11-06: decode M1 and prefill M128,
  // K=N=2048, GS128, CPU reference + screen/confirm. This is not qualification
  // of other entries or a claim that every shape has identical performance.
  if (runtime == 71626375) {
    if (architecture == "gfx1150") return {"mmnb4-validated-a", 128};
    if (architecture == "gfx1151") return {"mmnb4-validated-b", 128};
    if (architecture == "gfx1152") return {"mmnb4-validated-c", 128};
  }
  // A deterministic longer screen for the measured driver/entry scope.
  // It is still shared across five groups; spread remains diagnostic.
  if (runtime == 71260601 && architecture == "gfx1170")
    return {"mmnb4-driver-screen", 64, 500.0, 1000.0};
  return normal;
}

}  // namespace timing
}  // namespace hipdnn_ep_test
