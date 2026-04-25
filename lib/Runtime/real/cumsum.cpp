/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_cumsum.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

static int hipdnn_ep_to_hip_dtype_cumsum(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

extern "C" int wrap_cumsum(RuntimeState *state, void *input, void *output,
                           int64_t outer, int64_t axis_size, int64_t inner,
                           int64_t data_type, int64_t exclusive,
                           int64_t reverse) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_cumsum: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_cumsum(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_cumsum: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cumsum: dtype=%s, outer=%lld, axis=%lld, inner=%lld, "
      "excl=%lld, rev=%lld\n",
      hipdnn_ep_datatype_name(data_type), (long long)outer,
      (long long)axis_size, (long long)inner, (long long)exclusive,
      (long long)reverse);

  return hip_cumsum(stream, input, output, outer, axis_size, inner, hip_dtype,
                    static_cast<int>(exclusive), static_cast<int>(reverse));
}
