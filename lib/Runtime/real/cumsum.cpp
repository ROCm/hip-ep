/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// CumSum: y = cumulative-sum of x along `axis` (ONNX-14 attribute model:
// exclusive / reverse flags, axis as input tensor).
//
// Source: onnxruntime/core/providers/cuda/math/cumsum_impl.cu @ v1.22.2.
//
// The axis input is a GPU scalar; we D2H-read it once per call. The lowering
// already passes data_shape as a host int64 array, so the outer/axis/inner
// decomposition is done on the host without inspecting GPU tensors.
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

  void *stream = hipdnn_ep_state_get_stream(state);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // ONNX CumSum-14: axis is a 0-D scalar (single element) of int32 or
  // int64. The MLIR pipeline doesn't fold this constant for us, so we
  // synchronously D2H-read it. This adds one stall per CumSum call --
  // acceptable because the op typically occurs once or twice per graph.
  int64_t axis_value = 0;
  if (axis_dtype == HIPDNN_EP_DATATYPE_INT32) {
    int32_t a32 = 0;
    hipError_t err = hipMemcpyAsync(&a32, axis, sizeof(int32_t),
                                    hipMemcpyDeviceToHost, hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_cumsum: D2H axis (int32) failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    err = hipStreamSynchronize(hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_cumsum: stream sync after D2H axis failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    axis_value = static_cast<int64_t>(a32);
  } else if (axis_dtype == HIPDNN_EP_DATATYPE_INT64) {
    hipError_t err = hipMemcpyAsync(&axis_value, axis, sizeof(int64_t),
                                    hipMemcpyDeviceToHost, hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_cumsum: D2H axis (int64) failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    err = hipStreamSynchronize(hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_cumsum: stream sync after D2H axis failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
  } else {
    fprintf(stderr,
            "[REAL] wrap_cumsum: unsupported axis_dtype=%s(%lld) "
            "(supported: i32, i64)\n",
            hipdnn_ep_datatype_name(axis_dtype), (long long)axis_dtype);
    return -1;
  }

  // Normalize negative axis (idx in [-rank, rank)).
  if (axis_value < 0)
    axis_value += data_rank;
  if (axis_value < 0 || axis_value >= data_rank) {
    fprintf(stderr, "[REAL] wrap_cumsum: axis=%lld out of range [0, %lld)\n",
            (long long)axis_value, (long long)data_rank);
    return -1;
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < axis_value; ++d)
    outer *= data_shape[d];
  int64_t axis_size = data_shape[axis_value];
  int64_t inner = 1;
  for (int64_t d = axis_value + 1; d < data_rank; ++d)
    inner *= data_shape[d];

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cumsum: axis=%lld, outer=%lld, axis_size=%lld, "
      "inner=%lld, data_type=%s, excl=%lld, rev=%lld -> hip_cumsum\n",
      (long long)axis_value, (long long)outer, (long long)axis_size,
      (long long)inner, hipdnn_ep_datatype_name(data_type),
      (long long)exclusive, (long long)reverse);

  return hip_cumsum(stream, x, y, outer, axis_size, inner, hip_dtype,
                    exclusive ? 1 : 0, reverse ? 1 : 0);
}
