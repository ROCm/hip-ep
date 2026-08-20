/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// InstanceNormalization: y = scale * (x - mean) / sqrt(var + epsilon) + B
// Mean/var are computed per (N, C) over the spatial axes.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <string>

static int hipdnn_ep_to_hip_dtype_instance_norm(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

HIPDNN_EP_RT_EXPORT int wrap_instance_normalization(
    RuntimeState *state, void *input, void *scale, void *bias, void *output,
    int64_t n, int64_t c, int64_t spatial, int64_t data_type, float epsilon) {
  OP_PROFILE(
      "instancenorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lld", (long long)n, (long long)c,
                 (long long)spatial);
        return std::string(b);
      },
      state);

  if (!state || !input || !scale || !bias || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_instance_normalization: null required arg\n");
    return -1;
  }
  if (n <= 0 || c <= 0 || spatial <= 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_instance_normalization: empty shape n=%lld c=%lld "
        "spatial=%lld\n",
        (long long)n, (long long)c, (long long)spatial);
    return 0;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_instance_norm(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_instance_normalization: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_instance_normalization: n=%lld c=%lld spatial=%lld "
      "dtype=%lld eps=%e\n",
      (long long)n, (long long)c, (long long)spatial, (long long)data_type,
      (double)epsilon);

  return hip_instance_norm(stream, input, scale, bias, output, n, c, spatial,
                           epsilon, hip_dtype);
}
