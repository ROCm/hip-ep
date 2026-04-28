/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

int wrap_split(RuntimeState *state, void *data, void *output, int64_t axis,
               int64_t offset, int64_t input_axis_dim, int64_t output_axis_dim,
               int64_t inner_size, int64_t output_num_elements,
               int64_t element_size_bytes) {
  if (!state || !data || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_split: null argument\n");
    return -1;
  }
  if (axis < 0 || offset < 0 || input_axis_dim < 0 || output_axis_dim < 0 ||
      inner_size < 0 || output_num_elements < 0 || element_size_bytes <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_split: invalid argument values\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_split: axis=%lld offset=%lld input_axis_dim=%lld "
      "output_axis_dim=%lld inner_size=%lld output_num=%lld elem_size=%lld "
      "-> calling hip_split\n",
      (long long)axis, (long long)offset, (long long)input_axis_dim,
      (long long)output_axis_dim, (long long)inner_size,
      (long long)output_num_elements,
      (long long)element_size_bytes);

  return hip_split(stream, data, output, axis, offset, input_axis_dim,
                   output_axis_dim, inner_size, output_num_elements,
                   element_size_bytes);
}

