/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime for hip.orca_rmsnorm_l2 (fused ORCA RMSNorm-L2):
//   out[r,i] = weight[i] * x[r,i] / sqrt(sum_i x[r,i]^2)
// Exact fused form of the decode graph's mul(x,x)->reduce_sum->sqrt->
// reciprocal->mul->mul chain (weight absorbs 1/N; no epsilon). fp32.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

extern "C" int wrap_orca_rmsnorm_l2(
    RuntimeState *state,
    const void *input,
    const void *weight,
    void *output,
    int64_t outer,
    int64_t norm_size) {
  OP_PROFILE(
      "orca_rmsnorm_l2",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "outer=%lld H=%lld", (long long)outer,
                 (long long)norm_size);
        return std::string(b);
      },
      state);

  if (!state || !input || !weight || !output) {
    fprintf(stderr, "[REAL] wrap_orca_rmsnorm_l2: null argument\n");
    return -1;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "[REAL] wrap_orca_rmsnorm_l2: null stream\n");
    return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_orca_rmsnorm_l2: outer=%lld H=%lld\n",
                    (long long)outer, (long long)norm_size);

  return hip_rmsnorm_l2_fused(stream, input, weight, output, outer, norm_size);
}
