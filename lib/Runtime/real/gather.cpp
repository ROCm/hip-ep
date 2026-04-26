/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t element_size_bytes, int64_t pre_axis_size) {
  if (!state || !data || !indices || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_gather: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather: axis=%lld, data_num=%lld, indices_num=%lld, "
      "output_num=%lld, elem_size=%lld, pre_axis_size=%lld -> calling hip_gather\n",
      (long long)axis, (long long)data_num_elements,
      (long long)indices_num_elements, (long long)output_num_elements,
      (long long)element_size_bytes, (long long)pre_axis_size);

  int rc = hip_gather(stream, data, indices, output, axis, data_num_elements,
                      indices_num_elements, output_num_elements,
                      static_cast<int>(element_size_bytes), pre_axis_size);
  if (rc == 0)
    nan_trace_check("gather", output, output_num_elements);
  return rc;
}
