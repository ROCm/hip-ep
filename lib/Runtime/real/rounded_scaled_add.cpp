/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "rounded_scaled_add.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"

extern "C" int wrap_rounded_scaled_add_f16(
    RuntimeState *state, const void *base, const void *x, const void *y,
    void *output, int64_t bn, int64_t bc, int64_t bh, int64_t bw, int64_t xn,
    int64_t xc, int64_t xh, int64_t xw, int64_t yn, int64_t yc, int64_t yh,
    int64_t yw, int64_t on, int64_t oc, int64_t oh, int64_t ow,
    int64_t productOnLhs) {
  OP_PROFILE(
      "rounded_scaled_add", [&] { return std::to_string(ow); }, state);
  if (!state)
    return -1;
  const int64_t bs[] = {bn, bc, bh, bw}, xs[] = {xn, xc, xh, xw};
  const int64_t ys[] = {yn, yc, yh, yw}, os[] = {on, oc, oh, ow};
  return hip_rounded_scaled_add_f16(hipdnn_ep_state_get_stream(state), base, x,
                                    y, output, bs, xs, ys, os, productOnLhs);
}
