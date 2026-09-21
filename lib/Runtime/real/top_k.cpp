/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// Before: D2H memcpy of GPU K plus hipStreamSynchronize, then
//         hip_top_k(..., k_val, ...).
// After:  host `k` is the values memref dim at `axis` (passed from HipToLLVM).
//         OnnxToHip already materialized K to size tensor.empty.
int wrap_top_k(RuntimeState *state, void *x, void *values, void *indices,
               int64_t axis, int64_t largest, int64_t sorted, int64_t rank,
               const int64_t *x_shape, int64_t num_elements,
               int64_t element_size_bytes, int64_t k) {
  OP_PROFILE(
      "top_k",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld", (long long)rank,
                 (long long)axis);
        return std::string(b);
      },
      state);

  (void)num_elements;
  if (!state || !x || !values || !indices || !x_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null argument\n");
    return -1;
  }
  if (rank < 1 || rank > 8) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: rank must be in [1, 8]\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_top_k: axis=%lld, k=%lld, rank=%lld, largest=%lld, "
      "sorted=%lld -> hip_top_k\n",
      (long long)axis, (long long)k, (long long)rank, (long long)largest,
      (long long)sorted);

  return hip_top_k(stream, x, values, indices, axis, largest, sorted, rank,
                   x_shape, k, static_cast<int>(element_size_bytes));
}
