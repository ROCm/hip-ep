/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

static int arg_max_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_arg_max(RuntimeState *state, void *data, void *indices, int64_t axis,
                 int64_t keepdims, int64_t select_last_index, int64_t rank,
                 const int64_t *data_shape, int64_t data_type) {
  OP_PROFILE(
      "arg_max",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld:%s", (long long)rank,
                 (long long)axis, hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)keepdims;
  if (!state || !data || !indices || !data_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_arg_max: null argument\n");
    return -1;
  }

  int hip_dtype = arg_max_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_arg_max: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_arg_max: axis=%lld, keepdims=%lld, select_last=%lld, "
      "rank=%lld, dtype=%s -> hip_arg_max\n",
      (long long)axis, (long long)keepdims, (long long)select_last_index,
      (long long)rank, hipdnn_ep_datatype_name(data_type));

  return hip_arg_max(stream, data, indices, axis, select_last_index, rank,
                     data_shape, hip_dtype);
}
