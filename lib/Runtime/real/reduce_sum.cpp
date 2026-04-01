/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <hip/hip_runtime.h>

int wrap_reduce_sum(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t element_size_bytes,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state || !data || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_sum: null argument\n");
    return -1;
  }

  // Handle noop_with_empty_axes: if axes is empty and noop_with_empty_axes is
  // 1, copy input to output without reduction
  if (axes_num_elements == 0 && noop_with_empty_axes == 1) {
    void *stream = hipdnn_ep_state_get_stream(state);
    // Simple memcpy from data to output
    int64_t total_bytes = data_num_elements * element_size_bytes;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_reduce_sum: noop_with_empty_axes=1 with empty axes, "
        "copying %lld bytes\n",
        (long long)total_bytes);
    // Use hipMemcpyAsync to copy data to output
    hipError_t err =
        hipMemcpyAsync(output, data, total_bytes, hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(stream));
    return (err == hipSuccess) ? 0 : -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  int hip_dtype;
  switch (element_size_bytes) {
  case 8:
    hip_dtype = HIP_DTYPE_INT64;
    break;
  case 4:
    hip_dtype = HIP_DTYPE_INT32;
    break;
  default:
    RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_sum: unsupported element_size=%lld\n",
                      (long long)element_size_bytes);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_reduce_sum: data_num=%lld, output_num=%lld, "
      "axes_num=%lld, elem_size=%lld, keepdims=%lld, "
      "noop_with_empty_axes=%lld, dtype=%d -> calling hip_reduce_sum\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, (long long)element_size_bytes,
      (long long)keepdims, (long long)noop_with_empty_axes, hip_dtype);

  return hip_reduce_sum(stream, data, output, data_num_elements,
                        output_num_elements, hip_dtype);
}
