/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static const char *reduction_name(int64_t id) {
  switch (id) {
  case 0:
    return "none";
  case 1:
    return "add";
  case 2:
    return "mul";
  case 3:
    return "min";
  case 4:
    return "max";
  default:
    return "<unknown>";
  }
}

int wrap_scatter_elements(RuntimeState *state, void *data, void *indices,
                          void *updates, void *output, int64_t axis,
                          int64_t reduction_id, int64_t rank,
                          const int64_t *data_shape,
                          const int64_t *indices_shape, int64_t num_updates,
                          int64_t element_size_bytes,
                          int64_t indices_element_size_bytes) {
  OP_PROFILE(
      "scatter_elements",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "axis=%lld:%s:n=%lld", (long long)axis,
                 reduction_name(reduction_id), (long long)num_updates);
        return std::string(b);
      },
      state);

  if (!state || !data || !indices || !updates || !output || !data_shape ||
      !indices_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_scatter_elements: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_scatter_elements: axis=%lld, reduction=%s, rank=%lld, "
      "num=%lld -> hip_scatter_elements\n",
      (long long)axis, reduction_name(reduction_id), (long long)rank,
      (long long)num_updates);

  return hip_scatter_elements(stream, data, indices, updates, output, axis,
                              reduction_id, rank, data_shape, indices_shape,
                              num_updates, static_cast<int>(element_size_bytes),
                              static_cast<int>(indices_element_size_bytes));
}
