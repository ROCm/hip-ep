/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

template <typename T>
static int readDeviceScalar(void *stream, const void *src, T &dst) {
  auto hip_stream = static_cast<hipStream_t>(stream);
  hipError_t err =
      hipMemcpyAsync(&dst, src, sizeof(T), hipMemcpyDeviceToHost, hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] readDeviceScalar: hipMemcpyAsync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  // Synchronization is required here: we branch on host value `delta` to
  // preserve ORT's runtime delta==0 error behavior before launching hip_range.
  err = hipStreamSynchronize(hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[REAL] readDeviceScalar: hipStreamSynchronize failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  return 0;
}

int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype) {
  if (!state || !start || !limit || !delta || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_range: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  // ORT parity: Range must fail when delta == 0.
  int deltaCheck = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_INT16: {
    int16_t d = 0;
    deltaCheck = readDeviceScalar(stream, delta, d) || (d == 0);
    break;
  }
  case HIP_DTYPE_INT32: {
    int32_t d = 0;
    deltaCheck = readDeviceScalar(stream, delta, d) || (d == 0);
    break;
  }
  case HIP_DTYPE_INT64: {
    int64_t d = 0;
    deltaCheck = readDeviceScalar(stream, delta, d) || (d == 0);
    break;
  }
  case HIP_DTYPE_FLOAT32: {
    float d = 0.0f;
    // Keep exact zero check for ONNX/ORT parity.
    // ORT CPU Range rejects delta only when it is exactly zero.
    deltaCheck = readDeviceScalar(stream, delta, d) || (d == 0.0f);
    break;
  }
  case HIP_DTYPE_FLOAT64: {
    double d = 0.0;
    // Keep exact zero check for ONNX/ORT parity.
    // ORT CPU Range rejects delta only when it is exactly zero.
    deltaCheck = readDeviceScalar(stream, delta, d) || (d == 0.0);
    break;
  }
  default:
    fprintf(stderr, "[REAL] wrap_range: unsupported hip_dtype=%lld\n",
            (long long)hip_dtype);
    return -1;
  }
  if (deltaCheck) {
    fprintf(stderr, "[REAL] wrap_range: delta in Range operator can not be "
                    "zero!\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_range: output_num_elements=%lld, hip_dtype=%lld\n",
      (long long)output_num_elements, (long long)hip_dtype);

  return hip_range(stream, start, limit, delta, output, output_num_elements,
                   hip_dtype);
}
