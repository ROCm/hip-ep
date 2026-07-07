/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_gather_elements(RuntimeState *state, void *data, void *indices,
                         void *output, int64_t axis, int64_t rank,
                         const int64_t *data_shape,
                         const int64_t *indices_shape, int64_t num_elements,
                         int64_t element_size_bytes,
                         int64_t indices_element_size_bytes) {
  OP_PROFILE(
      "gather_elements",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:n=%lld", (long long)rank,
                 (long long)num_elements);
        return std::string(b);
      },
      state);

  if (!state || !data || !indices || !output || !data_shape ||
      !indices_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_gather_elements: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather_elements: axis=%lld, rank=%lld, num=%lld -> "
      "hip_gather_elements\n",
      (long long)axis, (long long)rank, (long long)num_elements);

  return hip_gather_elements(
      stream, data, indices, output, axis, rank, data_shape, indices_shape,
      num_elements, static_cast<int>(element_size_bytes),
      static_cast<int>(indices_element_size_bytes));
}
