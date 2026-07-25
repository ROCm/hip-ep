/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype) {
  if (!state || !start || !limit || !delta || !output) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_range: null argument (state=%p start=%p limit=%p "
        "delta=%p output=%p output_num_elements=%lld hip_dtype=%lld)\n",
        (void *)state, start, limit, delta, output,
        (long long)output_num_elements, (long long)hip_dtype);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_range: output_num_elements=%lld, hip_dtype=%lld\n",
      (long long)output_num_elements, (long long)hip_dtype);

  void *deviceErrorFlag = hipdnn_ep_state_get_error_flag_device_ptr(state);
  return hip_range(stream, start, limit, delta, output, output_num_elements,
                   hip_dtype, deviceErrorFlag);
}
