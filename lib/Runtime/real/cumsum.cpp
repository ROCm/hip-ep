/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// CumSum: y = cumulative-sum of x along `axis` (ONNX-14 attribute model:
// exclusive / reverse flags, axis as input tensor).
//
// Source: onnxruntime/core/providers/cuda/math/cumsum_impl.cu @ v1.22.2.
//
// The axis input is a GPU scalar. Reading it back to pick the launch
// geometry would drain the stream on every call, so the kernel loads it and
// decomposes the (host-known, statically shaped) data_shape around it
// instead; see hip_cumsum for how the grid is sized without knowing the
// axis. This wrapper only checks host-visible metadata.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

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

  (void)num_elements;

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

  // ONNX CumSum-14: axis is a 0-D scalar (single element) of int32 or
  // int64. Only its width is needed here; the kernel loads the value and
  // does the outer/axis_size/inner decomposition itself.
  int axis_is_int64;
  if (axis_dtype == HIPDNN_EP_DATATYPE_INT32) {
    axis_is_int64 = 0;
  } else if (axis_dtype == HIPDNN_EP_DATATYPE_INT64) {
    axis_is_int64 = 1;
  } else {
    fprintf(stderr,
            "[REAL] wrap_cumsum: unsupported axis_dtype=%s(%lld) "
            "(supported: i32, i64)\n",
            hipdnn_ep_datatype_name(axis_dtype), (long long)axis_dtype);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cumsum: rank=%lld, data_type=%s, excl=%lld, rev=%lld "
      "-> hip_cumsum\n",
      (long long)data_rank, hipdnn_ep_datatype_name(data_type),
      (long long)exclusive, (long long)reverse);

  return hip_cumsum(hipdnn_ep_state_get_stream(state), x, y, data_shape,
                    static_cast<int>(data_rank), axis, axis_is_int64, hip_dtype,
                    exclusive ? 1 : 0, reverse ? 1 : 0,
                    hipdnn_ep_state_get_error_flag_device_ptr(state));
}
