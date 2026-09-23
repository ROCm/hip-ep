/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// CumSum: y = cumulative-sum of x along `axis` (ONNX-14 attribute model:
// exclusive / reverse flags, axis as input tensor).
//
// Source: onnxruntime/core/providers/cuda/math/cumsum_impl.cu @ v1.22.2.
//
// Axis stays on the device. wrap_cumsum launches with num_elements threads
// (an upper bound on the slice count); hip_cumsum loads the axis scalar
// and splits data_shape into outer/axis_size/inner in shared memory.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int cumsum_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_cumsum(RuntimeState *state, void *x, void *axis, void *y,
                const int64_t *data_shape, int64_t data_rank,
                int64_t num_elements, int64_t data_type, int64_t axis_dtype,
                int64_t exclusive, int64_t reverse) {
  OP_PROFILE(
      "cumsum",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:%s%s%s", (long long)data_rank,
                 hipdnn_ep_datatype_name(data_type), exclusive ? ":excl" : "",
                 reverse ? ":rev" : "");
        return std::string(b);
      },
      state);

  if (!state || !x || !axis || !y || !data_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_cumsum: null argument\n");
    return -1;
  }
  if (data_rank <= 0) {
    fprintf(stderr, "[REAL] wrap_cumsum: invalid data_rank=%lld\n",
            (long long)data_rank);
    return -1;
  }

  int hip_dtype = cumsum_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cumsum: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  int axis_elem_bytes = 0;
  if (axis_dtype == HIPDNN_EP_DATATYPE_INT32)
    axis_elem_bytes = 4;
  else if (axis_dtype == HIPDNN_EP_DATATYPE_INT64)
    axis_elem_bytes = 8;
  else {
    fprintf(stderr,
            "[REAL] wrap_cumsum: unsupported axis_dtype=%s(%lld) "
            "(supported: i32, i64)\n",
            hipdnn_ep_datatype_name(axis_dtype), (long long)axis_dtype);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_cumsum: rank=%lld, num=%lld, data_type=%s, "
                    "excl=%lld, rev=%lld -> hip_cumsum\n",
                    (long long)data_rank, (long long)num_elements,
                    hipdnn_ep_datatype_name(data_type), (long long)exclusive,
                    (long long)reverse);

  return hip_cumsum(hipdnn_ep_state_get_stream(state), x, y, data_shape,
                    data_rank, num_elements, axis, axis_elem_bytes, hip_dtype,
                    exclusive ? 1 : 0, reverse ? 1 : 0,
                    hipdnn_ep_state_get_error_flag_device_ptr(state));
}
