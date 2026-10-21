/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "swiglu_f32.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"

extern "C" int wrap_swiglu_f32(RuntimeState *state, const void *gate,
                               const void *up, void *output, int64_t gn,
                               int64_t gc, int64_t gh, int64_t gw, int64_t un,
                               int64_t uc, int64_t uh, int64_t uw, int64_t on,
                               int64_t oc, int64_t oh, int64_t ow) {
  OP_PROFILE(
      "swiglu_f32", [&] { return std::to_string(ow); }, state);
  if (!state)
    return -1;
  const int64_t gateShape[] = {gn, gc, gh, gw};
  const int64_t upShape[] = {un, uc, uh, uw};
  const int64_t outShape[] = {on, oc, oh, ow};
  return hip_swiglu_f16_f32_bcast(hipdnn_ep_state_get_stream(state), gate, up,
                                  output, gateShape, upShape, outShape);
}
