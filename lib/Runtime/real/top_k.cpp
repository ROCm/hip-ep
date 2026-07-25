/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

int wrap_top_k(RuntimeState *state, void *x, void *k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *x_shape, int64_t num_elements,
               int64_t element_size_bytes) {
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
  if (!state || !x || !k || !values || !indices || !x_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null argument\n");
    return -1;
  }

  int64_t k_val = 0;
  void *stream = hipdnn_ep_state_get_stream(state);
  hipError_t err = hipMemcpy(&k_val, k, sizeof(int64_t), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: failed to read K: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: stream sync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_top_k: axis=%lld, k=%lld, rank=%lld, largest=%lld, "
      "sorted=%lld -> hip_top_k\n",
      (long long)axis, (long long)k_val, (long long)rank, (long long)largest,
      (long long)sorted);

  return hip_top_k(stream, x, values, indices, axis, largest, sorted, rank,
                   x_shape, k_val, static_cast<int>(element_size_bytes));
}
